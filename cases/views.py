from django.shortcuts import render


def cases(request):
    return render(request, "cases/index.html")
