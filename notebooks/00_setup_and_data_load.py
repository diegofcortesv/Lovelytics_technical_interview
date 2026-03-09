# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Setup & Data Load
# MAGIC **Fraud Copilot | Data Foundation**
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Creates the schema structure in Unity Catalog
# MAGIC 2. Loads fraud and purchase CSVs as Delta tables
# MAGIC 3. Loads financial documents and prepares them for RAG
# MAGIC 4. Validates everything is in place

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0.1 Configuration

# COMMAND ----------

CATALOG = "fraud_agent"
SCHEMA = "default"

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/data"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0.2 Create Schema & Volume
# MAGIC
# MAGIC Unity Catalog Volume is the recommended way to store files in Databricks.

# COMMAND ----------

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# Create a volume to upload raw files
spark.sql(f"""
    CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.data
    COMMENT 'Raw data files for Fraud Copilot'
""")

print(f"Schema {CATALOG}.{SCHEMA} ready")
print(f"Volume {CATALOG}.{SCHEMA}.data ready")
print(f"Upload your CSV and MD files to: {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0.3 Upload Files
# MAGIC
# MAGIC **Manual step:** Upload the following files to the Volume created above.
# MAGIC
# MAGIC Go to **Catalog → fraud_agent → default → data (Volume)** and upload:
# MAGIC - `fraud_dataset.csv`
# MAGIC - `product_purchase_dataset.csv`
# MAGIC - All `.md` files from `financial_documents/`
# MAGIC
# MAGIC Or use the cell below to upload programmatically if files are accessible.

# COMMAND ----------

# Verify files are uploaded
import os

try:
    files = dbutils.fs.ls(VOLUME_PATH)
    print("Files in volume:")
    for f in files:
        print(f"  {f.name} ({f.size:,} bytes)")
except Exception as e:
    print(f"Volume not accessible or empty: {e}")
    print(f"\nPlease upload files to: {VOLUME_PATH}")
    print("You can do this via Catalog UI → fraud_agent → default → data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0.4 Load Fraud Dataset to Delta Table

# COMMAND ----------

fraud_table = f"{CATALOG}.{SCHEMA}.fraud_dataset"

# Read CSV from volume
fraud_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(f"{VOLUME_PATH}/fraud_dataset.csv")
)

# Write as Delta table
(
    fraud_df.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(fraud_table)
)

# Verify
row_count = spark.table(fraud_table).count()
print(f"✓ {fraud_table}: {row_count} rows loaded")

# Show schema
spark.table(fraud_table).printSchema()

# COMMAND ----------

# Quick EDA on fraud dataset
display(
    spark.sql(f"""
        SELECT 
            COUNT(*) as total_transactions,
            SUM(CASE WHEN fraud = 1 THEN 1 ELSE 0 END) as fraud_count,
            SUM(CASE WHEN fraud = 0 THEN 1 ELSE 0 END) as legit_count,
            ROUND(AVG(transaction_amount), 2) as avg_amount,
            COUNT(DISTINCT customer_id) as unique_customers,
            ROUND(AVG(CAST(is_international AS DOUBLE)) * 100, 1) as intl_pct
        FROM {fraud_table}
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0.5 Load Purchase Dataset to Delta Table

# COMMAND ----------

purchase_table = f"{CATALOG}.{SCHEMA}.product_purchase_dataset"

purchase_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(f"{VOLUME_PATH}/product_purchase_dataset.csv")
)

(
    purchase_df.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(purchase_table)
)

row_count = spark.table(purchase_table).count()
print(f"{purchase_table}: {row_count} rows loaded")

spark.table(purchase_table).printSchema()

# COMMAND ----------

# Quick EDA on purchase dataset
display(
    spark.sql(f"""
        SELECT 
            membership_tier,
            COUNT(*) as count,
            ROUND(AVG(purchase_amount), 2) as avg_purchase,
            ROUND(AVG(loyalty_points), 0) as avg_loyalty
        FROM {purchase_table}
        GROUP BY membership_tier
        ORDER BY avg_purchase DESC
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0.6 Load Financial Documents for RAG
# MAGIC
# MAGIC We load the markdown documents into a Delta table with chunking for vector search.

# COMMAND ----------

import re

def chunk_markdown(content: str, doc_name: str, max_chars: int = 2000):
    """Chunk markdown by ## sections."""
    sections = re.split(r'\n(?=##\s)', content)
    chunks = []
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
        
        title_match = re.match(r'^##?\s+(.+)', section)
        section_title = title_match.group(1) if title_match else "Introduction"
        source = f"{doc_name} > {section_title}"
        
        if len(section) <= max_chars:
            chunks.append((section, source, section_title))
        else:
            paragraphs = section.split("\n\n")
            current = ""
            for para in paragraphs:
                if len(current) + len(para) > max_chars and current:
                    chunks.append((current.strip(), source, section_title))
                    current = para
                else:
                    current += "\n\n" + para
            if current.strip():
                chunks.append((current.strip(), source, section_title))
    
    return chunks


# Process all markdown files from volume
all_chunks = []
chunk_id = 0

try:
    md_files = [f for f in dbutils.fs.ls(VOLUME_PATH) if f.name.endswith('.md')]
    
    for md_file in sorted(md_files, key=lambda x: x.name):
        # Read file content
        content = dbutils.fs.head(md_file.path, 100000)  # max 100KB per file
        doc_name = md_file.name.replace('.md', '')
        
        chunks = chunk_markdown(content, doc_name)
        for text, source, section in chunks:
            all_chunks.append({
                "chunk_id": chunk_id,
                "content": text,
                "source": source,
                "doc_name": doc_name,
                "section": section,
                "char_count": len(text)
            })
            chunk_id += 1
        
        print(f"  ✓ {doc_name}: {len(chunks)} chunks")
    
    print(f"\nTotal: {len(all_chunks)} chunks from {len(md_files)} documents")

except Exception as e:
    print(f"Error reading MD files: {e}")
    print("Make sure .md files are uploaded to the volume")

# COMMAND ----------

# Save chunks as Delta table
if all_chunks:
    from pyspark.sql import Row
    
    chunks_table = f"{CATALOG}.{SCHEMA}.financial_docs_chunks"
    
    chunks_df = spark.createDataFrame([Row(**c) for c in all_chunks])
    
    (
        chunks_df.write
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(chunks_table)
    )
    
    count = spark.table(chunks_table).count()
    print(f"✓ {chunks_table}: {count} chunks saved")
    
    display(spark.table(chunks_table).select("chunk_id", "doc_name", "section", "char_count").limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0.7 Create Vector Search Index
# MAGIC

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
# 
VS_ENDPOINT_NAME = "fraud-copilot-vs-endpoint"
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.financial_docs_index"
EMBEDDING_MODEL = "databricks-gte-large-en"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.financial_docs_chunks"
# 
# # Create Vector Search endpoint
client = VectorSearchClient()
 
# # Check if endpoint exists, create if not
try:
    client.get_endpoint(VS_ENDPOINT_NAME)
    print(f"Endpoint {VS_ENDPOINT_NAME} already exists")
except:
    client.create_endpoint(
        name=VS_ENDPOINT_NAME,
        endpoint_type="STANDARD"
    )
    print(f"Created endpoint {VS_ENDPOINT_NAME}")
 
# # Enable Change Data Feed on source table (required for Delta Sync)
spark.sql(f"ALTER TABLE {SOURCE_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
 
# # Create the index
try:
    index = client.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT_NAME,
        index_name=VS_INDEX_NAME,
        source_table_name=SOURCE_TABLE,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="content",
        embedding_model_endpoint_name=EMBEDDING_MODEL,
    )
    print(f"Created index {VS_INDEX_NAME}")
    print("Index is syncing... this may take a few minutes.")
except Exception as e:
    if "already exists" in str(e):
        print(f"Index {VS_INDEX_NAME} already exists")
    else:
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0.8 Validation Summary

# COMMAND ----------

print("=" * 60)
print("FRAUD COPILOT - DATA FOUNDATION VALIDATION")
print("=" * 60)

tables = {
    "fraud_dataset": f"{CATALOG}.{SCHEMA}.fraud_dataset",
    "product_purchase": f"{CATALOG}.{SCHEMA}.product_purchase_dataset",
    "financial_docs": f"{CATALOG}.{SCHEMA}.financial_docs_chunks",
}

all_ok = True
for name, table in tables.items():
    try:
        count = spark.table(table).count()
        cols = len(spark.table(table).columns)
        print(f"  ✓ {name}: {count} rows, {cols} columns")
    except Exception as e:
        print(f"  ✗ {name}: MISSING - {e}")
        all_ok = False

print("=" * 60)
if all_ok:
    print("ALL TABLES READY - proceed to notebook 01")
else:
    print("ISSUES FOUND - check errors above")
print("=" * 60)