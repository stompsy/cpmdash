from __future__ import annotations

from datetime import date
from typing import Any, cast

from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin, PermissionRequiredMixin
from django.core.exceptions import PermissionDenied
from django.db.models import Count, Q, QuerySet
from django.forms import Form
from django.http import Http404, HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404
from django.urls import reverse_lazy
from django.utils import timezone
from django.views.generic import DetailView, ListView
from django.views.generic.edit import CreateView, DeleteView, UpdateView

from .forms import CaseStudyForm
from .models import CaseStudy, Tag


class CaseStudyListView(ListView):
    model = CaseStudy
    template_name = "blog/list.html"
    context_object_name = "case_studies"
    paginate_by = 9

    def get_queryset(self) -> QuerySet[CaseStudy]:
        now = timezone.now()
        qs = (
            CaseStudy.objects.filter(is_published=True)
            .filter(Q(published_at__lte=now) | Q(published_at__isnull=True))
            .order_by("-featured", "featured_rank", "-published_at", "-created_at")
        )
        tag = self.request.GET.get("tag")
        if tag:
            # Accept partial matches against tag name or slug
            qs = qs.filter(Q(tags__name__icontains=tag) | Q(tags__slug__icontains=tag)).distinct()
        return qs

    def get_template_names(self) -> list[str]:
        # Only use partial template for pagination (load more button)
        # Not for initial HTMX navigation from sidebar
        if self.request.headers.get("HX-Request") == "true" and self.request.GET.get("page"):
            return ["blog/_cards.html"]
        template_name = self.template_name or "blog/list.html"
        return [template_name]

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        ctx: dict[str, Any] = super().get_context_data(**kwargs)
        now = timezone.now()
        base_qs = Tag.objects.filter(case_studies__is_published=True).filter(
            Q(case_studies__published_at__lte=now) | Q(case_studies__published_at__isnull=True)
        )
        annotated_qs = cast(
            QuerySet[Tag],
            cast(Any, base_qs).annotate(case_count=Count("case_studies", distinct=True)),
        )
        ordered_qs = cast(QuerySet[Tag], cast(Any, annotated_qs).order_by("-case_count", "name"))
        ctx["popular_tags"] = cast(list[Tag], list(ordered_qs[:6]))
        ctx["active_tag"] = self.request.GET.get("tag", "")
        updated_on = date(2025, 10, 27)
        ctx.update(
            {
                "page_header_updated_at": updated_on,
                "page_header_updated_at_iso": updated_on.isoformat(),
            }
        )

        # Add "Create new case study" button for authenticated users with permission
        if self.request.user.is_authenticated and cast(Any, self.request.user).has_perm(
            "blog.add_casestudy"
        ):
            ctx["primary_cta_label"] = "Create new case study"
            ctx["primary_cta_url"] = reverse_lazy("blog:create")

        return ctx


class CaseStudyDetailView(DetailView):
    model = CaseStudy
    template_name = "blog/detail.html"
    context_object_name = "case_study"
    slug_field = "slug"
    slug_url_kwarg = "slug"

    def get_object(self, queryset: QuerySet[CaseStudy] | None = None) -> CaseStudy:
        now = timezone.now()
        filters = {"slug": self.kwargs.get("slug")}
        obj = get_object_or_404(CaseStudy, **filters)
        if (not obj.is_published or (obj.published_at and obj.published_at > now)) and not (
            self.request.user.is_authenticated and getattr(self.request.user, "is_staff", False)
        ):
            raise Http404()
        return obj

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        ctx: dict[str, Any] = super().get_context_data(**kwargs)
        obj: CaseStudy = ctx["case_study"]
        # Compute previous/next among published posts by published_at, fallback to created_at
        if obj.published_at:
            prev_qs = CaseStudy.objects.filter(
                is_published=True, published_at__lt=obj.published_at
            ).order_by("-published_at")
            next_qs = CaseStudy.objects.filter(
                is_published=True, published_at__gt=obj.published_at
            ).order_by("published_at")
        else:
            prev_qs = CaseStudy.objects.filter(
                is_published=True, created_at__lt=obj.created_at
            ).order_by("-created_at")
            next_qs = CaseStudy.objects.filter(
                is_published=True, created_at__gt=obj.created_at
            ).order_by("created_at")
        ctx["prev_post"] = prev_qs.first()
        ctx["next_post"] = next_qs.first()
        return ctx


class TagListView(ListView):
    model = Tag
    template_name = "blog/tags.html"
    context_object_name = "tags"

    def get_queryset(self) -> QuerySet[Tag]:
        now = timezone.now()
        published_filter = Q(case_studies__is_published=True) & (
            Q(case_studies__published_at__lte=now) | Q(case_studies__published_at__isnull=True)
        )
        annotated = cast(
            QuerySet[Tag],
            cast(Any, Tag.objects).annotate(
                case_count=Count("case_studies", filter=published_filter, distinct=True)
            ),
        )
        ordered = cast(QuerySet[Tag], cast(Any, annotated).order_by("-case_count", "name"))
        return ordered

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        ctx: dict[str, Any] = super().get_context_data(**kwargs)
        ctx["total_case_studies"] = CaseStudy.objects.filter(is_published=True).count()
        return ctx


class TagDetailView(ListView):
    template_name = "blog/tag_detail.html"
    context_object_name = "case_studies"
    paginate_by = 9

    def get_queryset(self) -> QuerySet[CaseStudy]:
        now = timezone.now()
        self.tag: Tag = get_object_or_404(Tag, slug=self.kwargs["slug"])
        return (
            CaseStudy.objects.filter(is_published=True)
            .filter(Q(published_at__lte=now) | Q(published_at__isnull=True))
            .filter(tags=self.tag)
            .order_by("-featured", "featured_rank", "-published_at", "-created_at")
        )

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        ctx: dict[str, Any] = super().get_context_data(**kwargs)
        ctx["tag"] = self.tag
        queryset = cast(QuerySet[CaseStudy] | None, ctx.get("case_studies"))
        paginator = ctx.get("paginator")
        ctx["case_count"] = (
            paginator.count if paginator else queryset.count() if queryset is not None else 0
        )
        now = timezone.now()
        published_filter = Q(case_studies__is_published=True) & (
            Q(case_studies__published_at__lte=now) | Q(case_studies__published_at__isnull=True)
        )
        related_base = Tag.objects.exclude(pk=self.tag.pk)
        related_annotated = cast(
            QuerySet[Tag],
            cast(Any, related_base).annotate(
                case_count=Count("case_studies", filter=published_filter, distinct=True)
            ),
        )
        related_ordered = cast(
            QuerySet[Tag], cast(Any, related_annotated).order_by("-case_count", "name")
        )
        ctx["related_tags"] = list(related_ordered[:6])
        return ctx

    def get_template_names(self) -> list[str]:
        if self.request.headers.get("HX-Request") == "true":
            return ["blog/_cards.html"]
        template_name = self.template_name or "blog/tag_detail.html"
        return [template_name]


class CaseStudyPermissionMixin(LoginRequiredMixin, PermissionRequiredMixin):
    raise_exception = False

    def handle_no_permission(self) -> HttpResponse:  # type: ignore[override]
        request = cast(HttpRequest, self.request)  # type: ignore[attr-defined]
        if not request.user.is_authenticated:
            return LoginRequiredMixin.handle_no_permission(self)
        raise PermissionDenied


class CaseStudyCreateView(CaseStudyPermissionMixin, CreateView):
    template_name = "blog/form.html"
    permission_required = "blog.add_casestudy"
    form_class = CaseStudyForm
    model = CaseStudy

    def form_valid(self, form: CaseStudyForm) -> HttpResponse:
        if not form.instance.author:
            form.instance.author = self.request.user
        response = super().form_valid(form)
        messages.success(self.request, "Case study created successfully.")
        return response

    def get_success_url(self) -> str:
        assert self.object is not None
        return self.object.get_absolute_url()


class CaseStudyUpdateView(CaseStudyPermissionMixin, UpdateView):
    template_name = "blog/form.html"
    permission_required = "blog.change_casestudy"
    slug_field = "slug"
    slug_url_kwarg = "slug"
    form_class = CaseStudyForm
    model = CaseStudy

    def form_valid(self, form: CaseStudyForm) -> HttpResponse:
        if not form.instance.author:
            form.instance.author = self.request.user
        response = super().form_valid(form)
        messages.success(self.request, "Case study updated successfully.")
        return response

    def get_success_url(self) -> str:
        assert self.object is not None
        return self.object.get_absolute_url()


class CaseStudyDeleteView(CaseStudyPermissionMixin, DeleteView):
    template_name = "blog/confirm_delete.html"
    permission_required = "blog.delete_casestudy"
    slug_field = "slug"
    slug_url_kwarg = "slug"
    success_url = reverse_lazy("blog:list")
    model = CaseStudy
    # Explicitly annotate to reconcile DeletionMixin/BaseDetailView attribute typing
    object: CaseStudy | None

    def form_valid(self, form: Form) -> HttpResponse:
        messages.success(self.request, "Case study deleted.")
        return super().form_valid(form)
