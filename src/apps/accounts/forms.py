from __future__ import annotations

from typing import Any

from allauth.account.forms import SignupForm
from django import forms
from django.core.exceptions import ValidationError

from .models import User


class DisabledSignupForm(SignupForm):
    """
    Signup form that always raises a validation error.
    Used to disable public account creation while keeping the URL accessible.
    """

    def clean(self) -> dict:
        """Prevent any signup attempts."""
        raise ValidationError(
            "Public account registration is disabled. "
            "Please contact your system administrator for access."
        )


class ProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = [
            "first_name",
            "last_name",
            "email",
            "bio",
            "avatar",
        ]
        widgets = {
            "first_name": forms.TextInput(
                attrs={
                    "class": "w-full rounded-md border border-white/15 bg-white/5 px-3 py-2 text-white placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-brand-400/60"
                }
            ),
            "last_name": forms.TextInput(
                attrs={
                    "class": "w-full rounded-md border border-white/15 bg-white/5 px-3 py-2 text-white placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-brand-400/60"
                }
            ),
            "email": forms.EmailInput(
                attrs={
                    "class": "w-full rounded-md border border-white/15 bg-white/5 px-3 py-2 text-white placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-brand-400/60"
                }
            ),
            "bio": forms.Textarea(
                attrs={
                    "rows": 4,
                    "class": "w-full rounded-md border border-white/15 bg-white/5 px-3 py-2 text-white placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-brand-400/60",
                }
            ),
        }

    def clean_email(self) -> str:
        # Do not allow email changes via this form yet; keep existing value.
        if self.instance and self.instance.pk:
            return self.instance.email
        return self.cleaned_data.get("email", "").strip()

    def clean_avatar(self) -> Any:
        avatar = self.cleaned_data.get("avatar")
        if not avatar:
            return avatar
        # Validate content type
        valid_content_types = {"image/jpeg", "image/png", "image/gif", "image/webp"}
        content_type = getattr(avatar, "content_type", None)
        if content_type and content_type not in valid_content_types:
            raise forms.ValidationError(
                "Unsupported image type. Please upload JPEG, PNG, GIF, or WEBP."
            )
        # Validate file size (e.g., 5MB max)
        max_bytes = 5 * 1024 * 1024
        if getattr(avatar, "size", 0) > max_bytes:
            raise forms.ValidationError("Avatar file is too large (max 5 MB).")
        return avatar
