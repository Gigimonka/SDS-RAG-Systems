# Data directory

Runtime documentation data is kept outside the Python package:

- `source/` — private Help&Manual projects used by the converter;
- `wikijs_export/` — generated Markdown imported by Wiki.js and indexed by RAG.

Both directories are intentionally ignored by Git. The converter expects the
unpacked Help&Manual project directly in `data/source/`:

```text
data/source/
├── Images/
├── Maps/
│   └── table_of_contents.xml
└── Topics/
```

`Baggage/` is optional. Damaged topic files are recorded in the conversion
report and do not stop conversion of the remaining topics.
