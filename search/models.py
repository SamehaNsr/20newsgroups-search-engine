from django.db import models
import pickle
from django.utils import timezone

# class BlogPost(models.Model):
#     title = models.CharField(max_length=200)
#     content = models.TextField()
#     author = models.ForeignKey('auth.User', on_delete=models.CASCADE)
#     created_at = models.DateTimeField(auto_now_add=True)
#     updated_at = models.DateTimeField(auto_now=True)
#     published = models.BooleanField(default=False)
#     view_count = models.IntegerField (default=0)
#     slug = models.SlugField(unique=True, max_length=255)
#     categories = models.ManyToManyField('Category')
#     image = models.ImageField(upload_to='blog_images/', blank=True, null=True)
#     metadata = models.JSONField(default=dict)
#     unique_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)

# Create your models here.
class Document(models.Model):
    document_id = models.IntegerField()
    document_text = models.TextField()
    category = models.CharField(max_length=200)
    def __str__(self):
        return f"Document {self.document_id}: {self.category}"

class ProcessedDocument(models.Model):
    document_id = models.IntegerField()
    document_text = models.TextField()
    category = models.CharField(max_length=200)
    document_embedding = models.BinaryField()

    def __str__(self):
        return f"Document {self.document_id}: {self.category}"

    # Optional: You can add a method to set and get embeddings, but it's not required.
    def set_embedding(self, embedding):
        self.document_embedding = pickle.dumps(embedding)

    def get_embedding(self):
        return pickle.loads(self.document_embedding)
    
class Query(models.Model):
    query_text = models.CharField(max_length=255)
    created_at = models.DateTimeField(default=timezone.now)
    results_count = models.IntegerField()
    
    def __str__(self):
        return f"Query: {self.query_text} at {self.created_at}"