# Local RAG Sources

Put local reference material here, then rebuild the index:

```powershell
D:\PycharmProjects\research\.venv\Scripts\python.exe scripts\build_knowledge_index.py
```

Supported now:

- `.txt`
- `.md` / `.markdown`
- `.csv`
- `.json`

Suggested materials:

- hERG medicinal chemistry notes
- literature summaries
- book excerpts you are allowed to use locally
- project-specific SAR notes
- assay protocol notes

PDF ingestion can be added after installing `pypdf` or `PyPDF2`.
