from typing import Any

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.http import HttpRequest

from .models import User


@admin.register(User)
class CustomUserAdmin(UserAdmin):
    def get_fieldsets(self, request: HttpRequest, obj: User | None = None) -> Any:
        fieldsets = list(super().get_fieldsets(request, obj))
        fieldsets.append(("Profile", {"fields": ("bio", "avatar")}))
        return fieldsets

    def get_list_display(self, request: HttpRequest) -> tuple[Any, ...]:
        base_display = tuple(super().get_list_display(request))
        return (*base_display, "bio")
