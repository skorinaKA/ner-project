# Распознавание сущностей в тексте (Named Entity Recognition) 

## 1.1. Введение в задачу NER

Определение задачи: выделение в тексте именованных сущностей (имена
людей, организации, локации, даты и т.д.) и их классификация по
предопределенным типам.

Значение и области применения: информационный поиск, семантический
анализ, машинный перевод, извлечение знаний, чат-боты, биомедицинские
исследования.

Named Entity Recognition (NER) - это ключевая задача обработки
естественного языка (Natural Language Processing, NLP), направленная на
автоматическое выявление и классификацию именованных сущностей в тексте.
Проще говоря, NER «читает» текст и отмечает в нём значимые объекты:
имена людей, названия организаций, географические наименования, даты,
денежные суммы и другие категории.

## 1.2. Как это работает

Процесс NER можно представить в виде последовательности шагов:

1\. **Токенизация** - разбиение текста на отдельные слова (токены).

2\. **Выделение признаков** - анализ контекста, морфологических
характеристик, капитализации и других маркеров.

3\. **Классификация** - присвоение каждому токену метки класса сущности
с помощью модели машинного обучения.

4\. **Объединение** - слияние смежных токенов в единую именованную
сущность (например, «Иван Иванович Петров» как одно имя).

Стандартные категории, которые обычно распознаёт NER‑система:

**PERSON** - имена людей (например, Анна Иванова);

**ORGANIZATION** - названия компаний, учреждений (Яндекс, Министерство
образования);

**LOCATION** - географические объекты (Москва, река Волга);

**DATE** - даты и временные периоды (5 марта 2025 года, XIX век);

**MONEY** - денежные суммы (1000 рублей, €50);

**PERCENT** - процентные значения (25 %);

**FACILITY** - сооружения, здания (Кремль, мост Золотые Ворота);

**GPE** - страны, регионы, города (Россия, \*Европа\*).

Для решения задачи NER применяются разные техники:

Правила и словари - использование заранее заданных шаблонов и списков
(например, список городов). Просто, но негибко.

Статистические модели - скрытые марковские модели (HMM), условные
случайные поля (CRF). Учитывают контекст и вероятности переходов между
метками.

Нейронные сети - LSTM, BiLSTM, трансформеры (BERT, RoBERTa). Позволяют
улавливать сложные семантические связи и работают с неочевидными
случаями.

Гибридные подходы - сочетание правил и машинного обучения для повышения
точности.

## 1.3 Применение

NER - фундамент для множества прикладных задач:

- Поиск информации - выделение ключевых сущностей для улучшения
  релевантности выдачи.

- Анализ социальных сетей - отслеживание упоминаний брендов, персон,
  событий.

- Извлечение знаний - построение графов знаний из текстов (например,
  «Иван Иванов работает в Яндексе»).

- Автоматизация документооборота - обработка счетов, контрактов, резюме.

- Чат‑боты и голосовые помощники - понимание намерений пользователя
  через распознавание сущностей.

- Биомедицина - выявление названий болезней, препаратов, генов в научных
  статьях.

## 1.4 Сложности и вызовы

Несмотря на прогресс, NER сталкивается с рядом проблем:

**Омонимия** - одно слово может обозначать разные сущности (\*«Apple»\*
как компания или фрукт).

**Нестандартные имена** - редкие имена, псевдонимы, аббревиатуры.

**Контекстная зависимость** - смысл сущности меняется в зависимости от
окружения.

**Многоязычность** - разные языки имеют свои морфологические и
синтаксические особенности.

**Сленг и опечатки** - неформальная речь и ошибки затрудняют
распознавание.

## 1.5 Современные тенденции

Сегодня развитие NER идёт в нескольких направлениях:

**Предобученные языковые модели** (BERT, XLNet) дают рекордную точность
за счёт глубокого понимания контекста.

**Нулевой shot‑learning** - распознавание сущностей без размеченных
данных для конкретной задачи.

**Мультимодальный NER** - анализ текста вместе с изображениями или аудио
для уточнения смысла.

**Адаптация под специальность** - тонкая настройка моделей для узких
областей (юриспруденция, медицина).

Таким образом, NER - это динамично развивающаяся область, которая делает
возможным «понимание» текста с помощью компьютеров и открывает двери для
интеллектуальных систем анализа данных.

## 1.5. Датасеты

1.  Few-NERD - это колоссальный набор данных с детальной ручной
    разметкой для распознавания именованных сущностей, который содержит
    8 обобщенных типов, 66 детализированных типов, 188 200 предложений,
    491 711 сущностей и 4 601 223 токена. Созданы три тестовые задачи:
    одна с учителем (Few-NERD (SUP)), а две другие - с малым количеством
    примеров (Few-NERD (INTRA) и Few-NERD (INTER)).

Схема Few-Nerd:

<img width="553" height="553" alt="image" src="https://github.com/user-attachments/assets/25364962-d1f2-4c4b-a3ec-ef4d0f3c45ad" />

В рамках курсовой работы будет использован датасет Few-NERD

Также, можно использовать и другие датасеты:

- [OntoNotes5](https://huggingface.co/datasets/tner/ontonotes5)

- [CoNLL03](https://huggingface.co/datasets/eriktks/conll2003)

- [CoNLL++](https://huggingface.co/predibase/conllpp)

- [MultiNERD](https://huggingface.co/datasets/Babelscape/multinerd) -
  есть датасет на русском языке

## 1.6 Энкодеры

Также, можно использовать и другие энкодеры:

- [prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny)

- [prajjwal1/bert-mini](https://huggingface.co/prajjwal1/bert-mini)

- [prajjwal1/bert-small](https://huggingface.co/prajjwal1/bert-small)

- [prajjwal1/bert-medium](https://huggingface.co/prajjwal1/bert-medium)

- [bert-base-cased](https://huggingface.co/bert-base-cased)

- [bert-large-cased](https://huggingface.co/bert-large-cased)

- [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)

- [bert-base-multilingual-uncased](https://huggingface.co/bert-base-multilingual-uncased)

- [roberta-base](https://huggingface.co/roberta-base)[roberta-large](https://huggingface.co/roberta-large)

- [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)

- [xlm-roberta-large](https://huggingface.co/xlm-roberta-large)

# 3. Метрики оценивания

Training loss:

<img width="554" height="317" alt="image" src="https://github.com/user-attachments/assets/fd37a24d-8fc0-4b7d-a135-d6e9d448f295" />



График train/loss демонстрирует монотонное снижение с \~0.045 до почти
0.000 к 14 эпохе. Это ожидаемо: модель хорошо подстраивается под
обучающие данные, ошибка на тренировке уменьшается

Темп обучения:

<img width="554" height="321" alt="image" src="https://github.com/user-attachments/assets/58059218-3a76-4cc3-9047-cfabe4969b31" />


График показывает плавное снижение скорости обучения (с \~1.5e-4 до
близкого к нулю). Это соответствует использованию планировщика и должно
способствовать стабильной сходимости, но, судя по метрикам, не
предотвратило переобучение

Градиент:

<img width="554" height="316" alt="image" src="https://github.com/user-attachments/assets/3bf3f8c8-2078-4869-a85e-a4d32c58a752" />


По такому графику трудно оценить ситуацию с градиентами

F1:

<img width="554" height="316" alt="image" src="https://github.com/user-attachments/assets/55e9c123-dbaf-48b3-83bf-3fc199178714" />


Полнота:

<img width="554" height="314" alt="image" src="https://github.com/user-attachments/assets/158ea1b0-7096-410c-89bf-93bdf8152c58" />


Точность:

<img width="554" height="314" alt="image" src="https://github.com/user-attachments/assets/72c3d7cf-f587-4505-a455-a84fa42cd50f" />


Validation loss:

<img width="554" height="315" alt="image" src="https://github.com/user-attachments/assets/94311278-2dfc-47b9-9603-53cdf43907ed" />


График train/loss демонстрирует монотонное снижение с \~0.05 до почти
0.000 к 14 эпохе. Это ожидаемо, так как модель хорошо подстраивается под
обучающие данные, ошибка на тренировке уменьшается

Рост validation loss с каждой эпохой при обучении модели машинного
обучения может указывать на переобучение.

Precision вырос, но recall остаётся нестабильным. Возможно, модель стала
слишком осторожной, пропуская часть сущностей ради точности. Возможно
стоит проверить настроить порог для принятия решений.

Модель успешно обучается, но после 7 эпохи начинает переобучаться (eval
loss растёт, метрики колеблются). Оптимальная эпоха - 14, где
достигается наилучший F1 (0.6588). Мы имеем не очень хорошие показатели
метрики F1.

# 4. Реализация решения

## 4.1. Стек технологий

### 4.1.1 Сервис по сбору метрик

Язык программирования: Python 3.12.12.

Библиотеки для машинного обучения: PyTorch

Датасет: kaggle, huggingface

Библиотеки для работы с трансформерами: transformers (предобученные
модели, токенизаторы), span_marker, spacy.

Библиотеки для обработки данных: pandas, numpy.

Подсчет метрик: evaluate.

Библиотеки для сбора метрик и визуализации: wandb - (Weights & Biases).

Контейнеризация: docker

Среда разработки: jupyter

### 4.1.2 Веб-сервис распознававания сущностей в тексте

Бэкэнд: Python3.11, fastapi, spaCy

Интерфейс: HTML, Bootstrap (Jinja2)

Визуализация: spaCy displacy, Matplotlib

API: /api/entity

Контейнеризация: docker

## 4.2 Архитектура решения

**Структура сервиса по сбору метрик:**

├── notebooks/ \# папка с .ipynb файлами

├── requirements.txt \# Python зависимости

├──
[docker-compose.yml](https://github.com/skorinaKA/ner-project/blob/main/docker-compose.yml)
#docker-compose установка

└── Dockerfile \# Docker установка

**Структура проекта веб-сервиса:**

├── app.py \# fastapi бэкэнд

├── templates/

│ └── index.html \# Интерфейс

├── requirements.txt \# Python зависимости

└── Dockerfile \# Docker установка

## 4.3 Инструкция по установке

### 4.3.1. Локально

1.  Установите Anaconda или Miniconda

2.  Добавьте Conda в переменные среды:

Windows:

> Если установлено для пользователя добавьте следующие каталоги в
> переменные среды вашего пользователя:
>
> C:\\Users\\YourUsername\\anaconda3
>
> C:\\Users\\YourUsername\\anaconda3\\Scripts
>
> C:\\Users\\YourUsername\\anaconda3\\Library\\bin
>
> Если установлено для всех пользователей добавьте следующие каталоги в
> переменные среды вашей системы:
>
> C:\\ProgramData\\Anaconda3
>
> C:\\ProgramData\\Anaconda3\\Scripts
>
> C:\\ProgramData\\Anaconda3\\Library\\bin

Linux/macOS:

> Добавьте следующую строку в файл конфигурации вашей оболочки
> (например, \~/.bashrc, \~/.bash_profile, \~/.zshrc):
>
> export PATH=\"/путь/к/anaconda3/bin:\$PATH\"
>
> \~ source/.bashrc \# или \~/.bash_profile, \~/.zshrc

3.  Создайте среду Conda:

conda create -n mlenv python=3.12 -y

conda activate mlenv

4.  Установите PyTorch с поддержкой CUDA:

Сначала проверьте версию вашего графического процессора и CUDA с помощью
nvidia-smi. Затем установите PyTorch с соответствующей версией CUDA с
помощью conda:

Этот метод обеспечивает установку CUDA и cuDNN через Conda

conda install cudatoolkit -c anaconda -y

conda install pytorch-cuda=12.4 -c pytorch -c nvidia -y

conda install pytorch torchvision torchaudio -c pytorch -c nvidia -y

5.  Проверьте правильность установки:

python -c \"import torch; print(torch.cuda.is_available())\"

Если на выходе получилось True, значит, PyTorch успешно установлен с
поддержкой графического процессора.

Для деактивации среды Conda нужно:

conda deactivate mlenv

Чтобы удалить среду Conda нужно:

conda env remove -n mlenv

### 4.3.2. Через docker

git clone <https://github.com/skorinaKA/ner-project>

cd ner-project

Если на ПК есть Gpu от Nvidia, то:

docker-compose \--profile gpu up -d

В противном случае:

docker-compose \--profile cpu up -d

## 4.4 Примеры использования сервисов

### 4.4.1 wandb

1.  Заходим на <http://localhost:8080/> и создаем аккаунт в wandb.

2.  Нажимаем на Generate. Это скопирует API-ключ

<img width="620" height="200" alt="image" src="https://github.com/user-attachments/assets/74142c80-9325-4f2a-a2f1-19bff2eefdba" />


3.  Открываем терминал в jupyter (New-\>Terminal) и вводим:

<img width="554" height="295" alt="image" src="https://github.com/user-attachments/assets/b0bd9874-85e9-4691-a1d6-f1b1850206eb" />

    cd notebooks

    wandb login \--relogin \--host=http://wandb:8080

4.  Вставляем скопированный ранее API-ключ.

<img width="554" height="175" alt="image" src="https://github.com/user-attachments/assets/dee9abf0-0ccc-487a-a7a1-44b95604e2c4" />


### 4.4.2 jupyter

1.  Заходим на [http://localhost:8888/](http://localhost:8080/)

2.  Переходим в папку notebooks и запускаем различные примеры:

3.  В процессе обучения будет представлена ссылка в которой собираются
    метрики для wandb. Например:

<img width="554" height="107" alt="image" src="https://github.com/user-attachments/assets/c1ff92c9-d6a2-496c-8d74-2aa077bccd7a" />


    Важно! Если заходить с браузера, то нужно поменять в ссылке wandb на
    localhost, так как в контейнере две разные ссылки: для хоста и для
    другого контейнера

### 4.4.3 ner-app-spacy-fastapi

#### Описание

Это полноценное веб-приложение NLP для распознавания именованных
объектов (NER) с использованием spaCy и fastapi. Это приложение
позволяет пользователям вводить текст или загружать текстовые файлы,
выбирать определенные типы объектов и интерактивно визуализировать
именованные объекты. Поддерживает несколько языков.

git clone <https://github.com/skorinaKA/ner-app-spacy-fastapi.git>

cd ner-app-spacy-fastapi

docker build -t ner-app .

docker run -p 8000:8000 ner-app

Открываем <http://localhost:8000/>

#### Пример работы приложения:

<img width="553" height="348" alt="image" src="https://github.com/user-attachments/assets/971b1bf9-b50b-4e11-9bf6-c9a78b5a2990" />


<img width="554" height="419" alt="image" src="https://github.com/user-attachments/assets/40ed635d-bb51-4113-adea-07bcd9a71985" />


#### Доступные api:

POST /api/entity - json-запрос для распознавания текста. text - исходный
текст. model - выбранная модель.

Request Body (JSON):

{

\"text\": \"Apple was founded by Steve Jobs in California.\",

\"model\": \"en_core_web_sm\"

}

Response Example:

{

\"entities\": \[

{\"text\": \"Apple\", \"label\": \"ORG\", \"start_char\": 0,
\"end_char\": 5},

{\"text\": \"Steve Jobs\", \"label\": \"PERSON\", \"start_char\": 21,
\"end_char\": 31},

{\"text\": \"California\", \"label\": \"GPE\", \"start_char\": 35,
\"end_char\": 45}

\]

}

POST /update - скачивает новую модель spacy

Request Body (JSON):

{

\"model\": \"en_core_web_sm\"

}

Response Example:

{

\"success\": \"Model downloaded\"

}

GET /models - возвращает список доступных моделей spacy

Response Example:

{

\"xx_ent_wiki_sm\": \"3.8.0\",

\"en_core_web_sm\": \"3.8.0\"

}

# Вывод

Я реализовал сервис по распознаванию именнованных сущностей в тексте с
возможностью обучения, инференса, тестирования, логирования метрик и
воспроизводимого развёртывания через Docker. А также было реалтизовано
веб-приложение NLP для распознавания именованных объектов (NER) с
использованием spaCy и fastapi

Низкие абсолютные значения метрики F1 (около 0.65) могут говорить о:

- проблемах датасета

- недостаточном размере модели

- необходимости дополнительной предобработки данных (например,
  использование BERT/RoBERTa с NER-разметкой дает высокие результаты).

- Необходимости дообучения с другими гиперпараметрами.

# Список литературы

1.  \[Электронный ресурс\] - <http://ningding97.github.io/fewnerd>

2.  \[Электронный ресурс\] -
    <http://www.kaggle.com/datasets/nbroad/fewnerd>

3.  \[Электронный ресурс\] -
    <https://huggingface.co/docs/transformers/model_doc/roberta>

4.  \[Электронный ресурс\] -
    <https://huggingface.co/docs/transformers/model_doc/xlnet>

5.  \[Электронный ресурс\] -
    <https://huggingface.co/datasets/DFKI-SLT/few-nerd>

6.  \[Электронный ресурс\] - <https://github.com/skorinaKA/ner-project>

7.  \[Электронный ресурс\] -
    <https://github.com/skorinaKA/ner-app-spacy-fastapi>
