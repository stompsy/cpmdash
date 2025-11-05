"""
Custom allauth adapter to disable public signups.
"""

from allauth.account.adapter import DefaultAccountAdapter
from django.contrib.auth.models import AbstractBaseUser
from django.http import HttpRequest, HttpResponseForbidden


class NoSignupAccountAdapter(DefaultAccountAdapter):
    """
    Custom adapter that disables signup functionality.
    Users must be created through Django admin.
    """

    def is_open_for_signup(self, request: HttpRequest) -> bool:
        """
        Override to disable public signup completely.
        """
        return False

    def respond_user_inactive(
        self, request: HttpRequest, user: AbstractBaseUser
    ) -> HttpResponseForbidden:
        """
        Custom response for inactive users.
        """
        return HttpResponseForbidden("This account has been deactivated.")
