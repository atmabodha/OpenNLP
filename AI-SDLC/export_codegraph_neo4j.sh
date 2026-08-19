#!/bin/bash

# ==============================================================================
# CODEGRAPH TO NEO4J EXPORT SCRIPT
# ==============================================================================
#
# WHAT IS CODEGRAPH?
#
# https://github.com/colbymchenry/codegraph
#
# CodeGraph indexes a codebase locally to build a semantic knowledge graph.
# It parses source files into:
#   - Nodes: Classes, methods, functions, and files.
#   - Edges: Relationships like calls, imports, extends, and references.
# It operates completely locally via a SQLite database (.codegraph/codegraph.db)
# with zero external API calls or persistent AI agents required.
#
# PURPOSE OF THIS SCRIPT:
# 1. Run a one-time CodeGraph index without starting background daemons.
# 2. Extract the generated 'nodes' and 'edges' tables to CSV files.
# 3. Clean up the .codegraph/ directory to leave no footprint on disk.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

echo "🔍 Generating CodeGraph (One-time run)..."
# CODEGRAPH_NO_DAEMON=1 prevents the background file-watching daemon from running
CODEGRAPH_NO_DAEMON=1 npx @colbymchenry/codegraph init

# Check if SQLite3 is installed
if ! command -v sqlite3 &> /dev/null; then
    echo "❌ Error: sqlite3 is not installed. Please install sqlite3 to export the database."
    exit 1
fi

echo "📦 Exporting nodes to nodes.csv..."
sqlite3 .codegraph/codegraph.db -csv -header "SELECT * FROM nodes;" > nodes.csv

echo "📦 Exporting edges to edges.csv..."
sqlite3 .codegraph/codegraph.db -csv -header "SELECT * FROM edges;" > edges.csv

echo "🧹 Cleaning up CodeGraph directory..."
rm -rf .codegraph/

echo "✅ Done! 'nodes.csv' and 'edges.csv' are ready for Neo4j import."

# ==============================================================================
# NEO4J IMPORT & VISUALIZATION INSTRUCTIONS
# https://neo4j.com
# ==============================================================================
#
# 1. SETUP:
#    - Move 'nodes.csv' and 'edges.csv' into your Neo4j DBMS 'import' directory.
#      (In Neo4j Desktop: DBMS menu "..." -> Open Folder -> Import)
#    - Start the DBMS and open Neo4j Browser.
#
# 2. RUN CYPHER QUERIES (In Neo4j Browser):
#
#    Step A: Create unique constraint on Node IDs (for fast lookups)
#    --------------------------------------------------------------
#    CREATE CONSTRAINT FOR (n:Node) REQUIRE n.id IS UNIQUE;
#
#    Step B: Load Nodes
#    --------------------------------------------------------------
#    LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
#    CREATE (:Node {
#        id: row.id,
#        name: row.name,
#        kind: row.kind,
#        filePath: row.file_path
#    });
#
#    Step C: Load Edges (Relationships)
#    --------------------------------------------------------------
#    LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
#    MATCH (sourceNode:Node {id: row.source})
#    MATCH (targetNode:Node {id: row.target})
#    CREATE (sourceNode)-[:LINK {type: row.kind}]->(targetNode);
#
# 3. SAMPLE VISUALIZATION QUERIES:
#
#    - General graph overview (limited to 150 items):
#      MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 150;
#
#    - Find a specific symbol and its connections:
#      MATCH (n:Node {name: "MyFunctionOrClassName"})-[r]-(connected)
#      RETURN n, r, connected;
#
#    - Filter by specific node type (e.g., classes):
#      MATCH (n:Node {kind: "class"})-[r]->(m)
#      RETURN n, r, m LIMIT 50;
#
# 4. BLOOM
#    Node LINK Node
# ==============================================================================
