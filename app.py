from fastmcp import FastMCP
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI
import pandas as pd

# ====================== SETTINGS ======================
client = OpenAI()  # uses OPENAI_API_KEY from Railway

# ====================== FLEXIBLE SCHEMA ======================
class ExtractedRecord(BaseModel):
    resident_name: Optional[str] = None
    resident_id: Optional[str] = None
    report_date: str
    document_type: str
    source_file: str
    extracted_fields: Dict[str, Any]
    raw_snippet: Optional[str] = None
    ingested_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

# ====================== MCP SERVER ======================
mcp = FastMCP(
    name="Meridian Meadows SNF Universal Pipeline",
    instructions="Intelligently ingest ANY SNF PDF/Excel from ChatGPT uploads. Extract ONLY pertinent clinical/admin data."
)

# PostgreSQL connection
def get_db_conn():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    conn.set_session(autocommit=True)
    return conn

def init_db():
    conn = get_db_conn()
    conn.cursor().execute("""
        CREATE TABLE IF NOT EXISTS master_records (
            id TEXT PRIMARY KEY,
            resident_name TEXT,
            resident_id TEXT,
            report_date TEXT,
            document_type TEXT,
            source_file TEXT,
            extracted_fields JSONB,
            raw_snippet TEXT,
            ingested_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    conn.close()

init_db()

@mcp.tool()
async def ingest_files(file_paths: List[str], custom_instructions: str = "") -> str:
    """Ingest files uploaded from ChatGPT. Handles temporary /mnt/data paths automatically."""
    results = []
    conn = get_db_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    for path in file_paths:
        filename = os.path.basename(path).lower()
        print(f"Processing file: {path} → {filename}")  # debug
        
        try:
            # Convert to markdown (works with ChatGPT temp paths)
            if path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(path)
                md = df.to_markdown(index=False)
            else:
                import pymupdf4llm
                md = pymupdf4llm.to_markdown(path)
            
            md_trunc = md[:120000] if len(md) > 120000 else md

            # LLM extraction
            prompt = f"""You are an expert SNF data extractor.
Extract ONLY pertinent clinical and administrative information.
Ignore headers, footers, signatures, page numbers.

Document: {filename}
Extra instructions: {custom_instructions or "Extract residents, incidents, progress notes, medication orders, vitals, weights, assessments, movements, etc."}

Return valid JSON with this exact structure:
{{"records": [{{
  "resident_name": "...",
  "resident_id": "...",
  "report_date": "YYYY-MM-DD",
  "document_type": "...",
  "extracted_fields": {{... any relevant keys ...}}
}}]}}

Markdown content:
{md_trunc}"""

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            data = json.loads(response.choices[0].message.content)
            records = data.get("records", [])
            
            for rec in records:
                record_id = f"{filename}_{datetime.utcnow().timestamp():.0f}"
                cur.execute("""
                    INSERT INTO master_records 
                    (id, resident_name, resident_id, report_date, document_type, 
                     source_file, extracted_fields, raw_snippet)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """, (
                    record_id,
                    rec.get("resident_name"),
                    rec.get("resident_id"),
                    rec.get("report_date", datetime.now().date().isoformat()),
                    rec.get("document_type", "unknown"),
                    filename,
                    json.dumps(rec.get("extracted_fields", rec)),
                    md[:500]
                ))
            
            results.append(f"✅ {filename} → {len(records)} records extracted and saved")

        except Exception as e:
            error_msg = f"❌ Error processing {filename}: {str(e)}"
            print(error_msg)
            results.append(error_msg)
    
    cur.close()
    conn.close()
    return "\n".join(results) + "\n\n✅ Master database updated successfully."

@mcp.tool()
async def query_master_db(sql: str = "SELECT * FROM master_records ORDER BY ingested_at DESC LIMIT 20") -> str:
    """Query the master database."""
    conn = get_db_conn()
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df.to_markdown(index=False) if not df.empty else "No records found."

@mcp.tool()
async def update_record(record_id: str, updates: dict) -> str:
    """Edit any existing record."""
    conn = get_db_conn()
    cur = conn.cursor()
    set_clause = ", ".join([f"{k}=%s" for k in updates.keys()])
    values = list(updates.values()) + [record_id]
    cur.execute(f"UPDATE master_records SET {set_clause} WHERE id = %s", values)
    cur.close()
    conn.close()
    return f"✅ Record {record_id} updated."

if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
