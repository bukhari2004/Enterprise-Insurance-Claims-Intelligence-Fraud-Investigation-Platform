# Technical Presentation Outline (10-12 slides, 10 min + live demo)
1. Problem: 18 M claims/yr, fraud losses, slow manual verification (numbers from the brief)
2. Solution overview: one pipeline - intake -> documents -> vision -> fraud -> recommendation -> copilot
3. Architecture diagram (docs/ARCHITECTURE.md mermaid) + event chain
4. Document intelligence: OCR -> extraction -> cross-checks -> tamper analysis (show an inflated invoice finding)
5. Computer vision: damage severity -> cost bands; reused-photo hashing; YOLO11 training plan
6. Fraud engine: explainable rule points + Isolation Forest; score decomposition screenshot
7. Knowledge graph: ring discovered in seed data (7 customers, 2 bank accounts, 1 garage) - show network view
8. Recommendation engine & alert center: 6 actions, priority = risk x exposure
9. Explainable AI & compliance: decision card, audit log, human-in-the-loop
10. Scalability: Redis/Kafka consumers, partitioned Postgres, Neo4j, object storage (25 M claims)
11. Results on seeded scenarios + honest limitations (synthetic data, baseline CV) + roadmap/bonus
12. Live demo script: (a) dashboard KPIs (b) open ring claim -> explain (c) submit new inflated claim + upload estimate -> watch score change (d) ask copilot 3 questions
