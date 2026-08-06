import os
import sys
import json

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from training.annotate_gemini import annotate

result = annotate(
    ticket="I forgot my password.",
    category="Account",
    subcategory="Reset Password",
)

print(json.dumps(result, indent=4))