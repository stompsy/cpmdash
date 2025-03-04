from django.shortcuts import render


def cases(request):
    title = "City of Port Angeles FD - Community Paramedicine"
    description = "Bridging The Gaps With Evidence-based Solutions"
    context = {"title": title, "description": description}
    return render(request, "cases/index.html", context=context)
