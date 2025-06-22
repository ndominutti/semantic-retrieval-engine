retrieve_ids_examples = {
    "requestBody": {
        "content": {
            "application/json": {
                "examples": {
                    "default": {
                        "summary": "Send a query and retrieve top similar documents IDs",
                        "value": {"query": "turquoise pillow", "top_n": 10},
                    },
                }
            }
        }
    }
}


retrieve_docs_examples = {
    "requestBody": {
        "content": {
            "application/json": {
                "examples": {
                    "default": {
                        "summary": "Send a query and retrieve top similar documents",
                        "value": {"query": "turquoise pillow", "top_n": 10},
                    },
                }
            }
        }
    }
}
