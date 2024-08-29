import pandas as pd
from django.core.management.base import BaseCommand
import os
from django.conf import settings
from search.models import ProcessedDocument  # Replace 'myapp' with your actual app name
# from sentence_transformers import SentenceTransformer
# import pickle
import torch
import faiss

class Command(BaseCommand):
    help = 'Import preprocessed documents'

    def handle(self, *args, **kwargs):
        # Load the CSV data into a DataFrame
        documents_df_path = os.path.join(settings.BASE_DIR, 'corpus', 'new_df.csv')
        document_embeddings_path = os.path.join(settings.BASE_DIR, 'corpus', 'new_embeddings.pt')
        # model = SentenceTransformer(model_path)
        documents_df = pd.read_csv(documents_df_path)
        document_embeddings = torch.load(document_embeddings_path)

        # # Initialize FAISS index
        document_embeddings = document_embeddings.cpu().numpy()

        # Initialize FAISS index
        # dimension = document_embeddings.shape[1]  # Embedding dimension
        # index = faiss.IndexFlatL2(dimension)

        # Loop through the DataFrame and corresponding embeddings
        for idx, row in documents_df.iterrows():
            doc = ProcessedDocument(
                document_id=row['document_id'],
                document_text=row['document_text'],
                category=row['category']
            )
            # Get the corresponding embedding and serialize it
            # embedding = document_embeddings[idx].cpu().numpy()  # Convert tensor to numpy array
            # print(embedding)
            # doc.set_embedding(embedding)
              # Serialize the corresponding embedding
            embedding = document_embeddings[idx]
            doc.set_embedding(embedding)
            # print(f'laai : {doc.get_embedding()}')
            # Save the Document instance to the database
            doc.save()
