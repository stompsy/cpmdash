from django.urls import path

from .views import (
    CaseStudyCreateView,
    CaseStudyDeleteView,
    CaseStudyDetailView,
    CaseStudyListView,
    CaseStudyUpdateView,
    TagDetailView,
    TagListView,
)

app_name = "blog"

urlpatterns = [
    path("", CaseStudyListView.as_view(), name="list"),
    path("new/", CaseStudyCreateView.as_view(), name="create"),
    path("<slug:slug>/edit/", CaseStudyUpdateView.as_view(), name="update"),
    path("<slug:slug>/delete/", CaseStudyDeleteView.as_view(), name="delete"),
    path("<slug:slug>/", CaseStudyDetailView.as_view(), name="detail"),
    path("tags/", TagListView.as_view(), name="tag-list"),
    path("tags/<slug:slug>/", TagDetailView.as_view(), name="tag-detail"),
]
