from django.shortcuts import render
from django.http import HttpResponse


def health_check(request):
    return HttpResponse("OK")


def opshield(request):
    title = "PORT Referrals"
    description = "Case Studies - OP Shielding Hope"
    context = { "title": title, "description": description, }
    return render(request, "cases/opshieldinghope.html", context=context)


def user_profile(request):
    return render(request, "dashboard/profile.html")