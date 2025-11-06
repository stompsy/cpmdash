import datetime as dt

import pytest
from django.contrib.auth.models import Permission
from django.urls import reverse
from django.utils import timezone

from apps.blog.models import CaseStudy, Tag


@pytest.fixture
def authenticated_client(client, django_user_model):
    """Fixture providing an authenticated client to bypass GlobalLoginRequiredMiddleware."""
    user = django_user_model.objects.create_user("testuser", "test@example.com", "password123")
    client.force_login(user)
    return client


@pytest.mark.django_db
class TestBlogViews:
    def test_list_view_empty(self, authenticated_client):
        url = reverse("blog:list")
        resp = authenticated_client.get(url)
        assert resp.status_code == 200
        assert b"No case studies" in resp.content

    def test_list_and_detail_published(self, authenticated_client):
        post = CaseStudy.objects.create(
            title="Test Post",
            slug="test-post",
            excerpt="Short intro",
            content="Line 1\n\nLine 2 with some extra words to expand count",
            is_published=True,
            published_at=timezone.now() - dt.timedelta(days=1),
        )
        t1 = Tag.objects.create(name="impact", slug="impact")
        t2 = Tag.objects.create(name="community", slug="community")
        post.tags.set([t1, t2])
        list_url = reverse("blog:list")
        resp = authenticated_client.get(list_url)
        assert resp.status_code == 200
        assert post.title.encode() in resp.content
        # word count tooltip presence (title attribute ending with 'words')
        assert b'title="' in resp.content and b'words"' in resp.content

        detail_url = reverse("blog:detail", kwargs={"slug": post.slug})
        resp2 = authenticated_client.get(detail_url)
        assert resp2.status_code == 200
        assert post.title.encode() in resp2.content
        assert b'title="' in resp2.content and b'words"' in resp2.content

    def test_detail_unpublished_404(self, authenticated_client):
        post = CaseStudy.objects.create(
            title="Draft",
            slug="draft-post",
            content="Hidden",
            is_published=False,
        )
        url = reverse("blog:detail", kwargs={"slug": post.slug})
        resp = authenticated_client.get(url)
        assert resp.status_code == 404

    def test_list_view_htmx_partial(self, authenticated_client):
        # create > paginate_by items to ensure pagination + OOB load-more container
        posts = []
        for i in range(10):
            posts.append(
                CaseStudy.objects.create(
                    title=f"HTMX Post {i}",
                    slug=f"htmx-post-{i}",
                    content="Body",
                    is_published=True,
                    published_at=timezone.now() - dt.timedelta(days=1),
                )
            )
        url = reverse("blog:list")
        resp = authenticated_client.get(url, **{"HTTP_HX_REQUEST": "true"})
        assert resp.status_code == 200
        assert b'id="cards-grid"' in resp.content
        # Ordered newest-first, so the last created should appear on page 1
        assert posts[-1].title.encode() in resp.content

        # should also OOB include load-more container markup
        assert b'id="load-more-container"' in resp.content

    def test_tag_detail_htmx_partial(self, authenticated_client):
        tag = Tag.objects.create(name="alpha", slug="alpha")
        post = CaseStudy.objects.create(
            title="Tag Post",
            slug="tag-post",
            content="Body",
            is_published=True,
            published_at=timezone.now() - dt.timedelta(days=1),
        )
        post.tags.set([tag])
        url = reverse("blog:tag-detail", kwargs={"slug": tag.slug})
        resp = authenticated_client.get(url, **{"HTTP_HX_REQUEST": "true"})
        assert resp.status_code == 200
        assert b'id="cards-grid"' in resp.content
        assert post.title.encode() in resp.content

    def test_list_view_partial_tag_search(self, authenticated_client):
        tag = Tag.objects.create(name="Harm Reduction", slug="harm-reduction")
        post = CaseStudy.objects.create(
            title="Naloxone Outreach",
            slug="naloxone-outreach",
            content="Body",
            is_published=True,
            published_at=timezone.now() - dt.timedelta(days=1),
        )
        post.tags.set([tag])

        other_post = CaseStudy.objects.create(
            title="Community Wellness",
            slug="community-wellness",
            content="Body",
            is_published=True,
            published_at=timezone.now() - dt.timedelta(days=2),
        )

        url = reverse("blog:list")
        resp = authenticated_client.get(url, {"tag": "harm"}, **{"HTTP_HX_REQUEST": "true"})
        assert resp.status_code == 200
        assert post.title.encode() in resp.content
        assert other_post.title.encode() not in resp.content


@pytest.fixture()
def manager_user(django_user_model):
    user = django_user_model.objects.create_user("manager", "manager@example.com", "password123")
    perms = Permission.objects.filter(
        codename__in=["add_casestudy", "change_casestudy", "delete_casestudy"],
        content_type__app_label="blog",
    )
    assert perms.count() == 3
    user.user_permissions.set(perms)
    return user


@pytest.mark.django_db
class TestCaseStudyCrud:
    def test_create_requires_permission(self, client, django_user_model):
        url = reverse("blog:create")
        # GlobalLoginRequiredMiddleware blocks anonymous users with 403
        resp = client.get(url)
        assert resp.status_code == 403
        # Authenticated users without permission also get 403
        user = django_user_model.objects.create_user("viewer", "viewer@example.com", "pass1234")
        client.force_login(user)
        resp = client.get(url)
        assert resp.status_code == 403

    def test_create_case_study(self, client, manager_user):
        tag = Tag.objects.create(name="Home Visits", slug="home-visits")
        client.force_login(manager_user)
        url = reverse("blog:create")
        payload = {
            "title": "Field follow-up on chronic care",
            "slug": "",
            "excerpt": "Short abstract",
            "content": "Detailed clinical narrative",
            "is_published": "on",
            "published_at": timezone.now().strftime("%Y-%m-%dT%H:%M"),
            "featured": "",
            "featured_rank": "0",
            "tags": [tag.pk],
        }
        resp = client.post(url, payload, follow=True)
        assert resp.status_code == 200
        assert resp.templates, resp.templates
        template_names = [tmpl.name for tmpl in resp.templates if getattr(tmpl, "name", None)]
        form = resp.context.get("form") if hasattr(resp.context, "get") else None
        if form is None and isinstance(resp.context, list):
            for ctx in resp.context:
                if isinstance(ctx, dict) and ctx.get("form") is not None:
                    form = ctx["form"]
                    break
        if form is not None:
            assert not form.errors, form.errors
        assert CaseStudy.objects.count() == 1, {
            "templates": template_names,
            "cases": list(CaseStudy.objects.values_list("title", "slug")),
        }
        case = CaseStudy.objects.get(title="Field follow-up on chronic care")
        assert case.slug == "field-follow-up-on-chronic-care"
        assert case.author == manager_user
        assert list(case.tags.all()) == [tag]

    def test_update_and_delete_case_study(self, client, manager_user):
        case = CaseStudy.objects.create(
            title="Initial title",
            slug="initial-title",
            content="Body",
            is_published=False,
        )
        client.force_login(manager_user)
        update_url = reverse("blog:update", kwargs={"slug": case.slug})
        update_payload = {
            "title": "Revised title",
            "slug": "revised-title",
            "excerpt": "Updated summary",
            "content": "Body",
            "is_published": "on",
            "published_at": "",
            "featured": "on",
            "featured_rank": "1",
            "tags": [],
        }
        resp = client.post(update_url, update_payload, follow=True)
        assert resp.status_code == 200
        case.refresh_from_db()
        assert case.title == "Revised title"
        assert case.slug == "revised-title"
        assert case.featured is True
        delete_url = reverse("blog:delete", kwargs={"slug": case.slug})
        resp = client.post(delete_url)
        assert resp.status_code == 302
        follow_resp = client.get(resp.headers["Location"], follow=True)
        assert follow_resp.status_code == 200
        assert not CaseStudy.objects.filter(pk=case.pk).exists()
