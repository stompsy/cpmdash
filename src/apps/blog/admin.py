from django.contrib import admin

from .models import CaseStudy, Tag


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ("name", "slug")
    prepopulated_fields = {"slug": ("name",)}


@admin.register(CaseStudy)
class CaseStudyAdmin(admin.ModelAdmin):
    list_display = ("title", "is_published", "published_at", "featured", "featured_rank")
    list_filter = ("is_published", "featured", "tags")
    search_fields = ("title", "excerpt", "content")
    filter_horizontal = ("tags",)
    prepopulated_fields = {"slug": ("title",)}
    ordering = ("-featured", "featured_rank", "-published_at")
