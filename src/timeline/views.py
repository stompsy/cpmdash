from django.shortcuts import render


def timeline(request):
    title = "Timeline"
    description = "This is a timeline page"
    context = {"title": title, "description": description}
    return render(request, "timeline/index.html", context=context)
