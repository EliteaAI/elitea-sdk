def search_format(items):
    results = []
    for (doc, score) in items:
        results.append({
            'page_content': doc.page_content,
            'metadata': doc.metadata,
            'score': score
        })
    return results