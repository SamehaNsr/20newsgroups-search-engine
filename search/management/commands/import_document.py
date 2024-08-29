import pandas as pd
from django.core.management.base import BaseCommand
from search.models import Document
import os
from django.conf import settings

class Command(BaseCommand):
    help = 'Import documents from CSV and generate embeddings'

    def handle(self, *args, **kwargs):
        # Load the CSV data into a DataFrame
        documents_df_path = os.path.join(settings.BASE_DIR, 'corpus', 'documents2.csv')
        documents_df = pd.read_csv(documents_df_path)

        # Iterate through the DataFrame and create Document instances
        for idx, row in documents_df.iterrows():
            # Create a new Document instance
            doc = Document(
                document_id=row['document_id'],
                document_text=row['document_text'],
                category=row['category']
            )
            # Save the Document instance to the database
            doc.save()
