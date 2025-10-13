from __future__ import annotations

from django.conf import settings
from django.db import models
from django.templatetags.static import static
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

    DEFAULT_HERO_IMAGES: dict[str, str] = {
        "access-to-care": "media/blog_defaults/access-to-care.png",
        "alcohol-use-disorder": "media/blog_defaults/alcohol-use-disorder.png",
        "behavioral-health": "media/blog_defaults/behavioral-health.png",
        "chronic-illnesses": "media/blog_defaults/chronic-illnesses.png",
        "education": "media/blog_defaults/education.png",
        "high-utilizers": "media/blog_defaults/high-utilizers.png",
        "home-fortification": "media/blog_defaults/home-fortification.png",
        "interagency-patients": "media/blog_defaults/interagency-patients.png",
        "medications": "media/blog_defaults/medications.png",
        "outreach": "media/blog_defaults/outreach.png",
        "patient-advocacy": "media/blog_defaults/patient-advocacy.png",
        "post-overdose-response-team": "media/blog_defaults/post-overdose-response-team.png",
        "procedures": "media/blog_defaults/procedures.png",
        "substance-use-disorder": "media/blog_defaults/substance-use-disorder.png",
        "wound-care": "media/blog_defaults/wound-care.png",
    }
    DEFAULT_HERO_FALLBACK: str = "media/blog_defaults/access-to-care.png"

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

    def _first_tag_slug(self) -> str | None:
        first_tag = self.tags.order_by("name").first()
        return first_tag.slug if first_tag else None

    def get_default_hero_image_path(self) -> str | None:
        slug = self._first_tag_slug()
        if not slug:
            return None
        return self.DEFAULT_HERO_IMAGES.get(slug)

    @property
    def hero_image_source(self) -> str | None:
        if self.hero_image:
            try:
                return self.hero_image.url
            except ValueError:
                # Storage backends raise ValueError when no file is present yet.
                pass

        default_path = self.get_default_hero_image_path()
        if default_path:
            return static(default_path)

        fallback_path = self.DEFAULT_HERO_FALLBACK
        return static(fallback_path) if fallback_path else None
