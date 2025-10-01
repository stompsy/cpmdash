from typing import Any

from django.contrib.auth.models import Group, Permission
from django.core.management.base import BaseCommand

ROLES = {
    "Coordinator": [
        "add_user",
        "change_user",
        "view_user",
    ],
    "Analyst": [
        "view_user",
    ],
}


class Command(BaseCommand):
    help = "Create default roles/groups and assign permissions"

    def handle(self, *args: Any, **options: Any) -> None:
        for role, perms in ROLES.items():
            group, _ = Group.objects.get_or_create(name=role)
            for codename in perms:
                try:
                    perm = Permission.objects.get(codename=codename)
                    group.permissions.add(perm)
                except Permission.DoesNotExist:
                    self.stdout.write(self.style.WARNING(f"Permission not found: {codename}"))
            self.stdout.write(self.style.SUCCESS(f"Ensured role: {role}"))
