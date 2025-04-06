import csv
import random

def get_terms_for_table():
    terms = []
    with open("./data/terms.csv", "r", encoding="utf-8") as f:
        cnt = 1
        for line in f.readlines()[1:]:
            term, definition, source = line.split(";")
            terms.append([cnt, term, definition])
            cnt += 1
    return terms


def write_term(new_term, new_definition):
    new_term_line = f"{new_term};{new_definition};user"
    with open("./data/terms.csv", "r", encoding="utf-8") as f:
        existing_terms = [l.strip("\n") for l in f.readlines()]
        title = existing_terms[0]
        old_terms = existing_terms[1:]
    terms_sorted = old_terms + [new_term_line]
    terms_sorted.sort()
    new_terms = [title] + terms_sorted
    with open("./data/terms.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(new_terms))


def get_terms_stats():
    db_terms = 0
    user_terms = 0
    defin_len = []

    with open("./data/terms.csv", "r", encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            term, defin, added_by = line.split(";")
            # Считаем количество букв в определении
            letter_count = sum(len(word) for word in defin.split())
            defin_len.append(letter_count)

            if "user" in added_by:
                user_terms += 1
            elif "db" in added_by:
                db_terms += 1

    # Вычисляем статистику
    stats = {
        "terms_all": db_terms + user_terms,
        "terms_own": db_terms,
        "terms_added": user_terms,
        "letters_avg": round(sum(defin_len) / len(defin_len), 2) if defin_len else 0,
        "letters_max": max(defin_len) if defin_len else 0,
        "letters_min": min(defin_len) if defin_len else 0
    }

    return stats

def get_random_term():
    with open("./data/terms.csv", mode='r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        terms = list(reader)

        # Пропускаем первую строку (заголовок)
        terms_without_header = terms[1:]

        return random.choice(terms_without_header)