from __future__ import annotations

from django.conf import settings
from django.db import models
from django.urls import reverse


class Tag(models.Model):
    name = models.CharField(max_length=50, unique=True)
    slug = models.SlugField(max_length=60, unique=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name

    def get_absolute_url(self) -> str:
        return reverse("blog:tag-detail", kwargs={"slug": self.slug})


class CaseStudy(models.Model):
    """Simple blog-style case study entry."""

    title = models.CharField(max_length=200)
    slug = models.SlugField(max_length=220, unique=True)
    excerpt = models.TextField(blank=True)
    content = models.TextField()
    hero_image = models.ImageField(upload_to="case_studies/", blank=True, null=True)
    published_at = models.DateTimeField(blank=True, null=True)
    is_published = models.BooleanField(default=False)
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        blank=True,
        null=True,
        related_name="case_studies",
    )
    tags: models.ManyToManyField[Tag, models.Model] = models.ManyToManyField(
        "Tag", related_name="case_studies", blank=True
    )
    featured = models.BooleanField(default=False)
    featured_rank = models.PositiveIntegerField(default=0, help_text="Lower numbers show first")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-featured", "featured_rank", "-published_at", "-created_at"]
        indexes = [
            models.Index(fields=["slug"]),
            models.Index(fields=["is_published", "published_at"]),
            models.Index(fields=["featured", "featured_rank"]),
        ]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.title

    def get_absolute_url(self) -> str:
        return reverse("blog:detail", kwargs={"slug": self.slug})
