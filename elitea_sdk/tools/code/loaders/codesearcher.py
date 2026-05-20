from ...chunkers.code.constants import get_programming_language, get_file_extension

def search_format(items):
    results = []
    for (doc, score) in items:
        language = get_programming_language(get_file_extension(doc.metadata.get("filename", "unknown")))
        method_name = doc.metadata.get("method_name", "text")
        formatted = doc.metadata.get("filename", "unknown") + " -> " + method_name + " (score: " + str(score) + ")"
        formatted += "\n\n```" + language.value + "\n" + doc.page_content + "\n```\n\n"
        results.append({
            'page_content': formatted,
            'metadata': doc.metadata,
            'score': score
        })
    return results