#!/usr/bin/env python3
"""Persistent ArxAgent research-session memory. No LLM/API calls."""

import argparse
import hashlib
import os
import uuid
from datetime import datetime, timezone

from falkordb import FalkorDB


def now():
    return datetime.now(timezone.utc).isoformat()


def stable_id(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


class SessionMemory:
    def __init__(self, graph_name="LLMStability"):
        db = FalkorDB(
            host=os.getenv("FALKORDB_HOST", "localhost"),
            port=int(os.getenv("FALKORDB_PORT", "6379")),
            password=os.getenv("FALKORDB_PASSWORD") or None,
        )
        self.graph = db.select_graph(graph_name)

    def query(self, cypher, params=None):
        return self.graph.query(cypher, params or {}).result_set

    def create(self, question):
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        sid = str(uuid.uuid4())
        timestamp = now()
        qid = stable_id(sid + "|" + question)
        self.query("""
            CREATE (s:ResearchSession {
                session_id: $sid, created_at: $ts,
                updated_at: $ts, status: 'active'
            })
            CREATE (q:ResearchQuestion {
                question_id: $qid, text: $question, created_at: $ts
            })
            CREATE (s)-[:ASKED]->(q)
        """, {"sid": sid, "qid": qid, "question": question,
              "ts": timestamp})
        return sid

    def record_claim(self, session_id, claim_id):
        ts = now()
        rows = self.query("""
            MATCH (s:ResearchSession {session_id: $sid})
            MATCH (c:Claim {claim_id: $cid})
            MERGE (s)-[r:INVESTIGATED]->(c)
            ON CREATE SET r.first_seen_at = $ts
            SET r.last_seen_at = $ts, s.updated_at = $ts
            RETURN s.session_id, c.claim_id, c.text
        """, {"sid": session_id, "cid": claim_id, "ts": ts})
        if not rows:
            raise ValueError("Session or claim not found")
        return rows[0]

    def add_open_question(self, session_id, question):
        question = question.strip()
        if not question:
            raise ValueError("Open question cannot be empty")
        oid = stable_id(session_id + "|" + question)
        ts = now()
        rows = self.query("""
            MATCH (s:ResearchSession {session_id: $sid})
            MERGE (q:OpenQuestion {question_id: $oid})
            ON CREATE SET q.text = $question, q.status = 'open',
                          q.created_at = $ts
            MERGE (s)-[:HAS_OPEN_QUESTION]->(q)
            SET s.updated_at = $ts
            RETURN q.question_id, q.text, q.status
        """, {"sid": session_id, "oid": oid, "question": question,
              "ts": ts})
        if not rows:
            raise ValueError("Session not found")
        return rows[0]

    def close_question(self, session_id, question):
        oid = stable_id(session_id + "|" + question.strip())
        rows = self.query("""
            MATCH (s:ResearchSession {session_id: $sid})
                  -[:HAS_OPEN_QUESTION]->
                  (q:OpenQuestion {question_id: $oid})
            SET q.status = 'resolved', q.resolved_at = $ts,
                s.updated_at = $ts
            RETURN q.question_id, q.text, q.status
        """, {"sid": session_id, "oid": oid, "ts": now()})
        if not rows:
            raise ValueError("Open question not found for this session")
        return rows[0]

    def show(self, session_id):
        sessions = self.query("""
            MATCH (s:ResearchSession {session_id: $sid})
            OPTIONAL MATCH (s)-[:ASKED]->(q:ResearchQuestion)
            RETURN s.session_id, s.created_at, s.updated_at, s.status,
                   collect(q.text) AS questions
        """, {"sid": session_id})
        if not sessions:
            return None
        claims = self.query("""
            MATCH (s:ResearchSession {session_id: $sid})
                  -[r:INVESTIGATED]->(c:Claim)
            OPTIONAL MATCH (p:Paper)-[:REPORTS]->(c)
            RETURN c.claim_id, c.text, p.paper_id, p.title,
                   r.first_seen_at, r.last_seen_at
            ORDER BY r.last_seen_at DESC
        """, {"sid": session_id})
        questions = self.query("""
            MATCH (s:ResearchSession {session_id: $sid})
                  -[:HAS_OPEN_QUESTION]->(q:OpenQuestion)
            RETURN q.question_id, q.text, q.status, q.created_at
            ORDER BY q.created_at
        """, {"sid": session_id})
        return {"session": sessions[0], "claims": claims,
                "open_questions": questions}

    def list_sessions(self, limit=10):
        return self.query("""
            MATCH (s:ResearchSession)
            OPTIONAL MATCH (s)-[:ASKED]->(q:ResearchQuestion)
            RETURN s.session_id, s.created_at, s.updated_at, s.status,
                   head(collect(q.text)) AS question
            ORDER BY s.updated_at DESC LIMIT $limit
        """, {"limit": limit})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("new")
    p.add_argument("--question", required=True)
    p = sub.add_parser("show")
    p.add_argument("--session", required=True)
    p = sub.add_parser("record-claim")
    p.add_argument("--session", required=True)
    p.add_argument("--claim", required=True)
    p = sub.add_parser("open-question")
    p.add_argument("--session", required=True)
    p.add_argument("--question", required=True)
    p = sub.add_parser("close-question")
    p.add_argument("--session", required=True)
    p.add_argument("--question", required=True)
    p = sub.add_parser("list")
    p.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()
    memory = SessionMemory()

    if args.command == "new":
        print("Session created:", memory.create(args.question))
    elif args.command == "record-claim":
        row = memory.record_claim(args.session, args.claim)
        print("Recorded claim:", row[1])
        print("Statement:", row[2])
    elif args.command == "open-question":
        row = memory.add_open_question(args.session, args.question)
        print("Open question:", row[1], "| Status:", row[2])
    elif args.command == "close-question":
        row = memory.close_question(args.session, args.question)
        print("Updated:", row[1], "| Status:", row[2])
    elif args.command == "list":
        for row in memory.list_sessions(max(1, min(args.limit, 50))):
            print(f"\nSession: {row[0]}\nStatus: {row[3]} | Updated: {row[2]}")
            print("Question:", row[4] or "(none)")
    elif args.command == "show":
        data = memory.show(args.session)
        if not data:
            raise SystemExit("Session not found")
        s = data["session"]
        print("Session:", s[0], "\nCreated:", s[1],
              "\nUpdated:", s[2], "\nStatus:", s[3])
        print("Original question:", ", ".join(s[4] or []) or "(none)")
        print("\nInvestigated claims:")
        for row in data["claims"]:
            print(f"- {row[0]} | {row[2]} — {row[3]}\n  {row[1]}")
        print("\nOpen and resolved questions:")
        for row in data["open_questions"]:
            print(f"- [{row[2]}] {row[1]}")
        if not data["claims"]:
            print("(No claims recorded yet)")
        if not data["open_questions"]:
            print("(No follow-up questions recorded)")


if __name__ == "__main__":
    main()
