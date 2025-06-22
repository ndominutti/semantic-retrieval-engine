rerank_ids_examples = {
    "requestBody": {
        "content": {
            "application/json": {
                "examples": {
                    "default": {
                        "summary": "Rerank given documents based on the query. Return relevant documents IDs",
                        "value": {
                            "query": "<QUERY>",
                            "documents": {
                                "docs": [
                                    {
                                        "product_id": 40707,
                                        "product_name": "<PRODUCT_NAME>",
                                        "product_class": "<PRODUCT_CLASS>",
                                        "category_hierarchy": "<PRODUCT_HIERARCHY>",
                                        "product_description": "<PRODUCT_DESCRIPTION>",
                                        "product_features": "<PRODUCT_FEATURES>",
                                        "rating_count": 10,
                                        "average_rating": 4.5,
                                        "review_count": 10,
                                        "score": 0.4,
                                    }
                                ]
                            },
                            "top_n": 10,
                        },
                    },
                }
            }
        }
    }
}


rerank_docs_examples = {
    "requestBody": {
        "content": {
            "application/json": {
                "examples": {
                    "default": {
                        "summary": "Rerank given documents based on the query. Return relevant documents",
                        "value": {
                            "query": "<QUERY>",
                            "documents": {
                                "docs": [
                                    {
                                        "product_id": 40707,
                                        "product_name": "<PRODUCT_NAME>",
                                        "product_class": "<PRODUCT_CLASS>",
                                        "category_hierarchy": "<PRODUCT_HIERARCHY>",
                                        "product_description": "<PRODUCT_DESCRIPTION>",
                                        "product_features": "<PRODUCT_FEATURES>",
                                        "rating_count": 10,
                                        "average_rating": 4.5,
                                        "review_count": 10,
                                        "score": 0.4,
                                    }
                                ]
                            },
                            "top_n": 10,
                        },
                    },
                }
            }
        }
    }
}
