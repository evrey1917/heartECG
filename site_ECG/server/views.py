from django.shortcuts import render
from server.forms import PatientForm


# def analyze_ekg(image_path):
#     # Пример обработки изображения (заглушка)
#     img = cv2.imread(image_path, 0)  # Читаем в оттенках серого
#     if img is None:
#         return "Ошибка обработки ЭКГ"
#     return "Здоров" if np.mean(img) > 100 else "Есть отклонения"


def upload_data(request):
    if request.method == "POST":
        form = PatientForm(request.POST, request.FILES)
        if form.is_valid():
            patient = form.save()
            patient.result = "Zdorov"
            patient.save()
            return render(request, "result.html", {"result": patient.result})
    else:
        form = PatientForm()
    return render(request, "upload.html", {"form": form})
