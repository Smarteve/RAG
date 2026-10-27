from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")
collection = "learning_vectors"

try:
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=3,
                distance=models.Distance.COSINE,
            ),
        )

    client.upsert(
        collection_name=collection,
        wait=True,
        points=[
            models.PointStruct(
                id=1,
                vector=[1.0, 0.0, 0.0],
                payload={"document_id": "handbook", "topic": "leave"},
            ),
            models.PointStruct(
                id=2,
                vector=[0.0, 1.0, 0.0],
                payload={"document_id": "handbook", "topic": "equipment"},
            ),
            models.PointStruct(
                id=3,
                vector=[0.9, 0.1, 0.0],
                payload={"document_id": "another-document"},
            ),
        ],
    )

    results = client.query_points(
        collection_name=collection,
        query=[1.0, 0.0, 0.0],
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value="handbook"),
                )
            ]
        ),
        limit=2,
        with_payload=True,
    ).points

    for point in results:
        print(point.id, point.score, point.payload)
finally:
    client.close()
