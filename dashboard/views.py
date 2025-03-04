from django.shortcuts import render


def dashboard(request):
    title = "Dashboard"
    description = "This is a Dashboard page"
    context = {"title": title, "description": description}
    return render(request, "dashboard/index.html", context=context)
