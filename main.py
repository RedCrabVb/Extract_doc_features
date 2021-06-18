import sys
import json
from extract_doc_features import extract_doc_features

sys.stdout.encoding
for param in sys.argv:
    try:
        d = extract_doc_features(param)
        print(json.dumps(d, ensure_ascii=False, indent=4))
    except Exception as e:
        print(e)
