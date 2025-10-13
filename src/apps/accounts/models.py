from django.contrib.auth.models import AbstractUser, Group, Permission
from django.db import models


class User(AbstractUser):
    """Custom user model.

    Extend here with additional fields later, e.g. department, phone, etc.
    """

    # Override to avoid clashes with auth.User
    groups = models.ManyToManyField(
        Group,
        verbose_name="groups",
        blank=True,
        help_text="The groups this user belongs to. A user will get all permissions granted to each of their groups.",
        related_name="accounts_user_set",
        related_query_name="accounts_user",
    )
    user_permissions = models.ManyToManyField(
        Permission,
        verbose_name="user permissions",
        blank=True,
        help_text="Specific permissions for this user.",
        related_name="accounts_user_set",
        related_query_name="accounts_user",
    )

    # Profile fields
    bio = models.TextField(blank=True, default="")
    # Save uploaded avatars under MEDIA_ROOT/user so they are served via MEDIA_URL.
    # Fallback avatars remain in the static pipeline under static/media/user.
    avatar = models.ImageField(upload_to="user/", blank=True, null=True)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.get_username()
