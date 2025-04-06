from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.core.cache import cache
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import UserPassesTestMixin
from . import terms_work


def index(request):
    return render(request, "index.html")

def terms_list(request):
    terms = terms_work.get_terms_for_table()
    return render(request, "term_list.html", context={"terms": terms})

@login_required
def add_term(request):
    if request.user.is_staff:  # Проверка на права администратора
        return render(request, "term_add.html")
    else:
        return redirect("index.html")


def send_term(request):
    if request.method == "POST":
        cache.clear()
        user_name = request.POST.get("name")
        new_term = request.POST.get("new_term", "")
        new_definition = request.POST.get("new_definition", "").replace(";", ",")
        context = {"user": user_name}
        if len(new_definition) == 0:
            context["success"] = False
            context["comment"] = "Описание должно быть не пустым"
        elif len(new_term) == 0:
            context["success"] = False
            context["comment"] = "Термин должен быть не пустым"
        else:
            context["success"] = True
            context["comment"] = "Ваш термин принят"
            terms_work.write_term(new_term, new_definition)
        if context["success"]:
            context["success-title"] = ""
        return render(request, "term_request.html", context)
    else:
        add_term(request)

@login_required
def show_stats(request):
    if request.user.is_staff:  # Проверка на права администратора
        stats = terms_work.get_terms_stats()
        return render(request, "stats.html", stats)
    else:
        return redirect("index.html")

@login_required
def flashcards(request):
    if request.method == "POST":
        user_definition = request.POST.get("new_definition", "").strip()
        correct_term, correct_definition, _ = request.session.get('current_term', (None, None, None))

        if user_definition.lower() == correct_definition.lower():
            result_message = "Успех! Ваш перевод правильный."
        else:
            result_message = f"Неудача! Правильное определение: {correct_definition}"

        # Получаем новый термин
        term, definition, _ = terms_work.get_random_term()
        request.session['current_term'] = (term, definition, _)  # Сохраняем новый термин в сессии

        return render(request, "flashcards.html", {
            "term": term,
            "result_message": result_message,
        })
    else:
        term, definition, _ = terms_work.get_random_term()
        request.session['current_term'] = (term, definition, _)  # Сохраняем термин в сессии
        return render(request, "flashcards.html", {
            "term": term,
            "result_message": None,
        })



def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('base')
        else:
            messages.error(request, 'Неверное имя пользователя или пароль')
    return render(request, 'login.html')


def logout_view(request):
    logout(request)
    return redirect('login')