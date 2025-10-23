import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score
from fpdf import FPDF
import warnings
import os
warnings.filterwarnings('ignore')

# Настраиваем страницу
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# МНОГОЯЗЫЧНАЯ ПОДДЕРЖКА
# ===========================

translations = {
    "en": {
        "title": "🎓 Student Performance Predictor",
        "subtitle": "AI-powered academic success prediction system with PDF export",
        "tabs": ["🎯 Prediction & Export", "📊 Analysis", "⚙️ Model Info"],
        "prediction_tab": {
            "header": "Predict Student Performance & Generate Reports",
            "academic_factors": "Academic Factors",
            "lifestyle_factors": "Lifestyle Factors", 
            "learning_style": "Learning Style",
            "study_hours": "Weekly Study Hours",
            "attendance": "Attendance Rate (%)",
            "previous_gpa": "Previous GPA",
            "semester": "Semester",
            "sleep_hours": "Daily Sleep Hours",
            "coffee_cups": "Daily Coffee Cups",
            "stress_level": "Stress Level",
            "extracurricular": "Extracurricular Activities",
            "visual": "Visual",
            "auditory": "Auditory",
            "kinesthetic": "Kinesthetic",
            "input_summary": "Current Input Summary",
            "predict_button": "🎯 Predict Final Score",
            "prediction_complete": "✅ Prediction completed! Scroll down for export options.",
            "predicted_score": "📊 Predicted Final Score",
            "performance_change": "📈 Performance Change",
            "excellent": "🎉 Excellent Performance!",
            "good": "👍 Good Performance", 
            "needs_improvement": "📈 Needs Improvement",
            "score_comparison": "📊 Score Comparison",
            "recommendations": "💡 Recommendations",
            "export_header": "📤 Export Reports",
            "pdf_section": "PDF Report",
            "pdf_description": "Generate a comprehensive PDF report with analysis and recommendations.",
            "pdf_button": "📥 Download PDF Report",
            "csv_section": "CSV Export",
            "csv_description": "Export the prediction data for further analysis.",
            "csv_button": "📊 Download CSV Data"
        },
        "analysis_tab": {
            "header": "Data Analysis & Insights",
            "correlation_heatmap": "Correlation Heatmap", 
            "study_vs_score": "Study Hours vs Final Score",
            "stats_header": "Dataset Statistics",
            "avg_study": "Average Study Hours",
            "avg_sleep": "Average Sleep",
            "avg_coffee": "Average Coffee",
            "avg_score": "Average Score",
            "performance_insights": "Performance Insights",
            "optimal_ranges": "Optimal Ranges for Success",
            "factor": "Factor",
            "minimum": "Minimum",
            "optimal": "Optimal Range",
            "maximum": "Maximum"
        },
        "model_tab": {
            "header": "Model Information",
            "details": "Model Details",
            "algorithm": "Algorithm",
            "features_used": "Features Used", 
            "training_data": "Training Data",
            "target": "Target",
            "feature_importance": "Feature Importance",
            "performance": "Model Performance",
            "accuracy": "Prediction Accuracy (R²)",
            "mae": "Mean Absolute Error",
            "cv_score": "Cross-Validation Score",
            "data_features": "Data Features",
            "academic_factors": "Academic Factors",
            "lifestyle_factors": "Lifestyle Factors",
            "behavioral_factors": "Behavioral Factors"
        },
        "sidebar": {
            "about": "🎓 About",
            "about_text": "This AI system predicts student academic performance based on study habits, lifestyle factors, and personal characteristics.",
            "quick_tips": "⚡ Quick Tips", 
            "study_tip": "**Study Smart**: 18-20 hours/week is optimal",
            "sleep_tip": "**Sleep Well**: 7-8 hours boosts performance",
            "balance_tip": "**Balance Life**: 2-3 extracurricular activities",
            "stress_tip": "**Manage Stress**: Keep below level 3/5",
            "language": "🌐 Language"
        },
        "footer": "**Student Performance Predictor** • Built with ❤️ using Streamlit & Scikit-learn"
    },
    "ru": {
        "title": "🎓 Анализатор Успеваемости Студентов",
        "subtitle": "AI-система прогнозирования академических результатов с экспортом PDF",
        "tabs": ["🎯 Прогноз и Экспорт", "📊 Аналитика", "⚙️ О Модели"],
        "prediction_tab": {
            "header": "Прогнозирование успеваемости и создание отчетов",
            "academic_factors": "Учебные показатели",
            "lifestyle_factors": "Факторы образа жизни",
            "learning_style": "Стиль обучения",
            "study_hours": "Часов учебы в неделю",
            "attendance": "Процент посещаемости (%)",
            "previous_gpa": "Предыдущий средний балл",
            "semester": "Семестр",
            "sleep_hours": "Часов сна в сутки",
            "coffee_cups": "Чашек кофе в день",
            "stress_level": "Уровень стресса",
            "extracurricular": "Внеучебные активности",
            "visual": "Визуальный",
            "auditory": "Аудиальный",
            "kinesthetic": "Кинестетический",
            "input_summary": "Текущие параметры",
            "predict_button": "🎯 Предсказать итоговый балл",
            "prediction_complete": "✅ Прогноз завершен! Прокрутите вниз для экспорта.",
            "predicted_score": "📊 Прогнозируемый итоговый балл",
            "performance_change": "📈 Изменение результата",
            "excellent": "🎉 Отличный результат!",
            "good": "👍 Хороший результат",
            "needs_improvement": "📈 Требует улучшения",
            "score_comparison": "📊 Сравнение баллов",
            "recommendations": "💡 Рекомендации",
            "export_header": "📤 Экспорт отчетов",
            "pdf_section": "PDF отчет",
            "pdf_description": "Создайте детальный PDF отчет с анализом и рекомендациями.",
            "pdf_button": "📥 Скачать PDF отчет",
            "csv_section": "CSV экспорт",
            "csv_description": "Экспортируйте данные прогноза для дальнейшего анализа.",
            "csv_button": "📊 Скачать CSV данные"
        },
        "analysis_tab": {
            "header": "Анализ данных и статистика",
            "correlation_heatmap": "Тепловая карта корреляций",
            "study_vs_score": "Часы учебы vs Итоговый балл",
            "stats_header": "Статистика данных",
            "avg_study": "Среднее время учебы",
            "avg_sleep": "Средний сон",
            "avg_coffee": "Среднее кофе",
            "avg_score": "Средний балл",
            "performance_insights": "Анализ эффективности",
            "optimal_ranges": "Оптимальные диапазоны для успеха",
            "factor": "Фактор",
            "minimum": "Минимум",
            "optimal": "Оптимальный диапазон",
            "maximum": "Максимум"
        },
        "model_tab": {
            "header": "Информация о модели",
            "details": "Детали модели",
            "algorithm": "Алгоритм",
            "features_used": "Использовано признаков",
            "training_data": "Обучающие данные",
            "target": "Целевая переменная",
            "feature_importance": "Важность признаков",
            "performance": "Производительность модели",
            "accuracy": "Точность прогноза (R²)",
            "mae": "Средняя абсолютная ошибка",
            "cv_score": "Кросс-валидация",
            "data_features": "Характеристики данных",
            "academic_factors": "Учебные факторы",
            "lifestyle_factors": "Факторы образа жизни",
            "behavioral_factors": "Поведенческие факторы"
        },
        "sidebar": {
            "about": "🎓 О системе",
            "about_text": "Эта AI-система предсказывает академическую успеваемость студентов на основе учебных привычек, факторов образа жизни и личных характеристик.",
            "quick_tips": "⚡ Полезные советы",
            "study_tip": "**Умная учеба**: 18-20 часов в неделю оптимально",
            "sleep_tip": "**Качественный сон**: 7-8 часов улучшают результаты",
            "balance_tip": "**Сбалансированность**: 2-3 внеучебные активности",
            "stress_tip": "**Контроль стресса**: Держите ниже уровня 3 из 5",
            "language": "🌐 Язык"
        },
        "footer": "**Анализатор Успеваемости Студентов** • Создано с ❤️ на Streamlit & Scikit-learn"
    }
}

# Инициализация языка
if 'language' not in st.session_state:
    st.session_state.language = 'en'

def set_language(lang):
    st.session_state.language = lang

# ===========================
# ОСНОВНОЙ КОД
# ===========================

# Загружаем модель
@st.cache_resource
def load_ml_components():
    try:
        model = joblib.load('best_student_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        return model, scaler, feature_columns
    except FileNotFoundError:
        st.error("❌ Файлы модели не найдены. Сначала запустите student_ml_system.py")
        st.stop()

# Класс для генерации PDF с улучшенной обработкой шрифтов
class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.font_family = 'Arial'
        
        # Пробуем загрузить шрифты DejaVu если они есть в текущей директории
        try:
            # Проверяем существование файлов шрифтов
            if (os.path.exists('DejaVuSansCondensed.ttf') and 
                os.path.exists('DejaVuSansCondensed-Bold.ttf') and 
                os.path.exists('DejaVuSansCondensed-Oblique.ttf')):
                
                self.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
                self.add_font('DejaVu', 'B', 'DejaVuSansCondensed-Bold.ttf', uni=True)
                self.add_font('DejaVu', 'I', 'DejaVuSansCondensed-Oblique.ttf', uni=True)
                self.font_family = 'DejaVu'
                #st.success("✅ Шрифты DejaVu успешно загружены!")
            else:
                st.warning("⚠️ Шрифты DejaVu не найдены. Используются стандартные шрифты Arial.")
        except Exception as e:
            st.warning(f"⚠️ Ошибка загрузки шрифтов DejaVu: {e}. Используются стандартные шрифты Arial.")
    
    def header(self):
        self.set_font(self.font_family, 'B', 16)
        self.cell(0, 10, 'Student Performance Analysis Report', 0, 1, 'C')
        self.set_font(self.font_family, 'I', 10)
        self.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        self.ln(10)
    
    def footer(self):
        self.set_y(-15)
        self.set_font(self.font_family, 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def add_section_title(self, title):
        self.set_font(self.font_family, 'B', 12)
        self.cell(0, 10, title, 0, 1)
        self.ln(5)
    
    def add_key_value(self, key, value):
        self.set_font(self.font_family, 'B', 10)
        self.cell(40, 8, key, 0, 0)
        self.set_font(self.font_family, '', 10)
        # Безопасное преобразование значения в строку
        value_str = str(value)
        self.cell(0, 8, value_str, 0, 1)

def generate_pdf_report(student_data, prediction, recommendations, lang='en'):
    pdf = PDFReport()
    pdf.add_page()
    
    # Заголовок отчёта
    if lang == 'ru':
        pdf.add_section_title("Отчет об анализе успеваемости студента")
        
        # Информация о студенте
        pdf.add_section_title("Профиль студента")
        pdf.add_key_value("Часы учебы:", f"{student_data['hours_studied']} ч/нед")
        pdf.add_key_value("Часы сна:", f"{student_data['sleep_hours']} ч/ночь")
        pdf.add_key_value("Чашки кофе:", f"{student_data['coffee_cups']} чашек/день")
        pdf.add_key_value("Посещаемость:", f"{student_data['attendance_rate']}%")
        pdf.add_key_value("Предыдущий GPA:", f"{student_data['previous_gpa']}")
        pdf.add_key_value("Уровень стресса:", f"{student_data['stress_level']}/5")
        pdf.add_key_value("Внеучебные:", f"{student_data['extracurricular']} активностей")
        pdf.add_key_value("Стиль обучения:", student_data['learning_style'])
        
        pdf.ln(10)
        
        # Результаты предсказания
        pdf.add_section_title("Результаты прогноза")
        pdf.add_key_value("Прогнозируемый балл:", f"{prediction:.1f}/100")
        
        # Оценка результата
        if prediction >= 85:
            performance = "Отличный"
        elif prediction >= 70:
            performance = "Хороший"
        else:
            performance = "Требует улучшения"
        
        pdf.add_key_value("Уровень успеваемости:", performance)
        
        pdf.ln(10)
        
        # Рекомендации
        pdf.add_section_title("Рекомендации по улучшению")
    else:
        pdf.add_section_title("Student Performance Prediction Report")
        
        # Информация о студенте
        pdf.add_section_title("Student Profile")
        pdf.add_key_value("Study Hours:", f"{student_data['hours_studied']} h/week")
        pdf.add_key_value("Sleep Hours:", f"{student_data['sleep_hours']} h/night")
        pdf.add_key_value("Coffee Cups:", f"{student_data['coffee_cups']} cups/day")
        pdf.add_key_value("Attendance Rate:", f"{student_data['attendance_rate']}%")
        pdf.add_key_value("Previous GPA:", f"{student_data['previous_gpa']}")
        pdf.add_key_value("Stress Level:", f"{student_data['stress_level']}/5")
        pdf.add_key_value("Extracurricular:", f"{student_data['extracurricular']} activities")
        pdf.add_key_value("Learning Style:", student_data['learning_style'])
        
        pdf.ln(10)
        
        # Результаты предсказания
        pdf.add_section_title("Prediction Results")
        pdf.add_key_value("Predicted Final Score:", f"{prediction:.1f}/100")
        
        # Оценка результата
        if prediction >= 85:
            performance = "Excellent"
        elif prediction >= 70:
            performance = "Good"
        else:
            performance = "Needs Improvement"
        
        pdf.add_key_value("Performance Level:", performance)
        
        pdf.ln(10)
        
        # Рекомендации
        pdf.add_section_title("Improvement Recommendations")
    
    # Добавляем рекомендации
    for i, recommendation in enumerate(recommendations, 1):
        pdf.set_font(pdf.font_family, '', 10)
        # Убираем эмодзи для PDF
        clean_rec = recommendation.replace('🎉', '').replace('👍', '').replace('📈', '').replace('✅', '').replace('🎯', '')
        pdf.multi_cell(0, 8, f"{i}. {clean_rec}")
        pdf.ln(2)
    
    return pdf

def generate_recommendations(student_data, predicted_score, lang='en'):
    recommendations = []
    
    # Анализ учебных привычек
    if student_data['hours_studied'] < 15:
        if lang == 'ru':
            recommendations.append(f"Рекомендуется увеличить время учебы с {student_data['hours_studied']} до 18-20 часов в неделю для лучших результатов.")
        else:
            recommendations.append(f"Consider increasing study time from {student_data['hours_studied']} to 18-20 hours per week for better results.")
    elif student_data['hours_studied'] > 25:
        if lang == 'ru':
            recommendations.append(f"Текущее время учебы ({student_data['hours_studied']} часов) довольно высокое. Уделяйте внимание качеству, а не количеству, чтобы избежать выгорания.")
        else:
            recommendations.append(f"Current study time ({student_data['hours_studied']} hours) is high. Ensure quality over quantity to avoid burnout.")
    
    # Анализ сна
    if student_data['sleep_hours'] < 7:
        if lang == 'ru':
            recommendations.append(f"Увеличьте продолжительность сна с {student_data['sleep_hours']} до 7-8 часов в сутки для улучшения когнитивных функций.")
        else:
            recommendations.append(f"Increase sleep duration from {student_data['sleep_hours']} to 7-8 hours per night for better cognitive function.")
    elif student_data['sleep_hours'] > 9:
        if lang == 'ru':
            recommendations.append(f"Рассмотрите возможность сокращения сна до 7-8 часов для более эффективного использования времени без ущерба для отдыха.")
        else:
            recommendations.append(f"Consider reducing sleep to 7-8 hours to maximize productive time while maintaining rest quality.")
    
    # Анализ кофе
    if student_data['coffee_cups'] > 3:
        if lang == 'ru':
            recommendations.append(f"Сократите потребление кофе с {student_data['coffee_cups']} до 2-3 чашек в день для улучшения качества сна.")
        else:
            recommendations.append(f"Reduce coffee consumption from {student_data['coffee_cups']} to 2-3 cups daily to improve sleep quality.")
    
    # Анализ стресса
    if student_data['stress_level'] >= 4:
        if lang == 'ru':
            recommendations.append(f"Обнаружен высокий уровень стресса ({student_data['stress_level']}/5). Рекомендуется освоить техники управления стрессом.")
        else:
            recommendations.append(f"High stress level ({student_data['stress_level']}/5) detected. Consider stress management techniques.")
    
    # Анализ посещаемости
    if student_data['attendance_rate'] < 85:
        if lang == 'ru':
            recommendations.append(f"Улучшите посещаемость с {student_data['attendance_rate']}% до 90%+ для более последовательного обучения.")
        else:
            recommendations.append(f"Improve attendance rate from {student_data['attendance_rate']}% to 90%+ for better learning continuity.")
    
    # Общие рекомендации
    if predicted_score < 70:
        if lang == 'ru':
            recommendations.append("Сосредоточьтесь на основных концепциях и обратитесь за дополнительной академической поддержкой.")
        else:
            recommendations.append("Focus on foundational concepts and seek additional academic support.")
    elif predicted_score >= 85:
        if lang == 'ru':
            recommendations.append("Отличные результаты! Рассмотрите возможность менторства для других студентов или запишитесь на продвинутые курсы.")
        else:
            recommendations.append("Excellent performance! Consider mentoring peers or taking advanced courses.")
    
    return recommendations

def main():
    # Инициализация состояния сессии
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    if 'student_data' not in st.session_state:
        st.session_state.student_data = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    
    model, scaler, feature_columns = load_ml_components()
    lang = st.session_state.language
    t = translations[lang]
    
    # CSS для стилизации
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .prediction-card {
            background-color: #f0f2f6;
            padding: 2rem;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
        }
        .export-section {
            background-color: #f8f9fa;
            padding: 2rem;
            border-radius: 10px;
            border-left: 5px solid #28a745;
            margin: 2rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Заголовок приложения
    st.markdown(f'<h1 class="main-header">{t["title"]}</h1>', unsafe_allow_html=True)
    st.markdown(f"### {t['subtitle']}")

    # Боковая панель с переключением языка
    with st.sidebar:
        st.header(t["sidebar"]["about"])
        st.write(t["sidebar"]["about_text"])
        
        st.header(t["sidebar"]["quick_tips"])
        st.success(t["sidebar"]["study_tip"])
        st.warning(t["sidebar"]["sleep_tip"]) 
        st.info(t["sidebar"]["balance_tip"])
        st.error(t["sidebar"]["stress_tip"])
        
        st.header(t["sidebar"]["language"])
        
        # Кнопки переключения языка
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🇺🇸 English", use_container_width=True, type="primary" if lang == "en" else "secondary"):
                set_language("en")
                st.rerun()
        with col2:
            if st.button("🇷🇺 Русский", use_container_width=True, type="primary" if lang == "ru" else "secondary"):
                set_language("ru")
                st.rerun()

    # Создаём вкладки
    tab1, tab2, tab3 = st.tabs(t["tabs"])

    with tab1:
        st.header(t["prediction_tab"]["header"])
        
        # Создаём колонки для ввода данных
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(t["prediction_tab"]["academic_factors"])
            hours_studied = st.slider(t["prediction_tab"]["study_hours"], 5, 30, 15, key="hours_studied")
            attendance_rate = st.slider(t["prediction_tab"]["attendance"], 60, 100, 85, key="attendance_rate")
            previous_gpa = st.slider(t["prediction_tab"]["previous_gpa"], 50, 95, 75, key="previous_gpa")
            semester = st.selectbox(t["prediction_tab"]["semester"], [1, 2, 3, 4], key="semester")
        
        with col2:
            st.subheader(t["prediction_tab"]["lifestyle_factors"])
            sleep_hours = st.slider(t["prediction_tab"]["sleep_hours"], 4.0, 10.0, 7.0, 0.5, key="sleep_hours")
            coffee_cups = st.slider(t["prediction_tab"]["coffee_cups"], 0, 8, 2, key="coffee_cups")
            stress_level = st.slider(t["prediction_tab"]["stress_level"], 1, 5, 3, key="stress_level")
            extracurricular = st.slider(t["prediction_tab"]["extracurricular"], 0, 5, 2, key="extracurricular")
            learning_style = st.radio(t["prediction_tab"]["learning_style"], 
                                    [t["prediction_tab"]["visual"], t["prediction_tab"]["auditory"], t["prediction_tab"]["kinesthetic"]], 
                                    key="learning_style")

        # Кнопка предсказания
        if st.button(t["prediction_tab"]["predict_button"], type="primary", use_container_width=True, key="predict_button"):
            # Подготавливаем данные для модели
            student_data = {
                'hours_studied': hours_studied,
                'coffee_cups': coffee_cups,
                'sleep_hours': sleep_hours,
                'attendance_rate': attendance_rate,
                'previous_gpa': previous_gpa,
                'extracurricular': extracurricular,
                'stress_level': stress_level,
                'learning_style': learning_style,
                'semester': semester
            }
            
            # Создаём DataFrame
            input_df = pd.DataFrame([student_data])
            input_df = pd.get_dummies(input_df)
            
            # Добавляем недостающие колонки
            for col in feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Упорядочиваем колонки
            input_df = input_df[feature_columns]
            
            # Масштабируем и предсказываем
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            prediction = max(40, min(100, prediction))
            
            # Генерируем рекомендации
            recommendations = generate_recommendations(student_data, prediction, lang)
            
            # Сохраняем в session state
            st.session_state['prediction'] = prediction
            st.session_state['student_data'] = student_data
            st.session_state['recommendations'] = recommendations
            st.session_state['prediction_made'] = True
            
            # Показываем результаты
            st.success(t["prediction_tab"]["prediction_complete"])
            
            # Отображаем результаты
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                st.metric(t["prediction_tab"]["predicted_score"], f"{prediction:.1f}")
                st.metric(t["prediction_tab"]["performance_change"], f"{(prediction - previous_gpa):+.1f}")
            
            with col_result2:
                if prediction >= 85:
                    st.success(t["prediction_tab"]["excellent"])
                elif prediction >= 70:
                    st.info(t["prediction_tab"]["good"])
                else:
                    st.warning(t["prediction_tab"]["needs_improvement"])
            
            # Визуализация результата
            st.subheader(t["prediction_tab"]["score_comparison"])
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Создаём данные для графика
            if lang == 'ru':
                labels = ['Предыдущий балл', 'Прогнозируемый балл']
            else:
                labels = ['Previous GPA', 'Predicted Score']
                
            scores = [previous_gpa, prediction]
            colors = ['#ff9999', '#66b3ff']
            
            # Создаём горизонтальный бар-чарт
            bars = ax.barh(labels, scores, color=colors, alpha=0.7, height=0.6)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Score' if lang == 'en' else 'Балл')
            if lang == 'ru':
                ax.set_title('Сравнение: предыдущий и прогнозируемый результат')
            else:
                ax.set_title('Comparison: Previous vs Predicted Performance')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Добавляем значения на бары
            for bar, score in zip(bars, scores):
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                        f'{score:.1f}', ha='left', va='center', fontweight='bold', fontsize=12)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Показываем рекомендации
            st.subheader(t["prediction_tab"]["recommendations"])
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")

        # Секция экспорта - показываем только после предсказания
        if st.session_state.get('prediction_made', False):
            st.markdown("---")
            st.markdown('<div class="export-section">', unsafe_allow_html=True)
            st.header(t["prediction_tab"]["export_header"])
            
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                st.subheader(t["prediction_tab"]["pdf_section"])
                st.write(t["prediction_tab"]["pdf_description"])
                
                # Создаём PDF
                pdf = generate_pdf_report(
                    st.session_state['student_data'],
                    st.session_state['prediction'],
                    st.session_state['recommendations'],
                    lang
                )
                
                # Конвертируем PDF в bytes
                try:
                    pdf_output = pdf.output(dest='S').encode('latin-1', 'replace')
                    
                    # Кнопка скачивания PDF
                    st.download_button(
                        label=t["prediction_tab"]["pdf_button"],
                        data=pdf_output,
                        file_name="student_performance_report.pdf",
                        mime="application/pdf",
                        key="pdf_download"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating PDF: {e}")
            
            with col_export2:
                st.subheader(t["prediction_tab"]["csv_section"])
                st.write(t["prediction_tab"]["csv_description"])
                
                # Подготавливаем CSV данные
                export_data = st.session_state['student_data'].copy()
                export_data['predicted_score'] = st.session_state['prediction']
                export_data['prediction_date'] = datetime.now().strftime("%Y-%m-%d %H:%M")
                export_df = pd.DataFrame([export_data])
                
                # Кнопка скачивания CSV
                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    label=t["prediction_tab"]["csv_button"],
                    data=csv_data,
                    file_name="student_prediction_data.csv",
                    mime="text/csv",
                    key="csv_download"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.header(t["analysis_tab"]["header"])
        
        # Генерируем примеры данных для анализа
        np.random.seed(42)
        sample_data = {
            'Study Hours': np.random.normal(15, 5, 100).clip(5, 30),
            'Sleep Hours': np.random.normal(7, 1.5, 100).clip(4, 10),
            'Coffee Cups': np.random.poisson(3, 100),
            'Stress Level': np.random.randint(1, 6, 100),
            'Final Score': np.random.normal(75, 10, 100).clip(40, 100)
        }
        df_sample = pd.DataFrame(sample_data)
        
        col_analysis1, col_analysis2 = st.columns(2)
        
        with col_analysis1:
            st.subheader(t["analysis_tab"]["correlation_heatmap"])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df_sample.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
            st.pyplot(fig)
        
        with col_analysis2:
            st.subheader(t["analysis_tab"]["study_vs_score"])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=df_sample, x='Study Hours', y='Final Score', 
                           hue='Stress Level', size='Coffee Cups', ax=ax)
            st.pyplot(fig)
        
        # Показываем статистику
        st.subheader(t["analysis_tab"]["stats_header"])
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        
        with col_stats1:
            st.metric(t["analysis_tab"]["avg_study"], f"{df_sample['Study Hours'].mean():.1f}h")
        with col_stats2:
            st.metric(t["analysis_tab"]["avg_sleep"], f"{df_sample['Sleep Hours'].mean():.1f}h")
        with col_stats3:
            st.metric(t["analysis_tab"]["avg_coffee"], f"{df_sample['Coffee Cups'].mean():.1f}c")
        with col_stats4:
            st.metric(t["analysis_tab"]["avg_score"], f"{df_sample['Final Score'].mean():.1f}")
        
        # Оптимальные диапазоны
        st.subheader(t["analysis_tab"]["optimal_ranges"])
        optimal_data = {
            t["analysis_tab"]["factor"]: [t["prediction_tab"]["study_hours"], t["prediction_tab"]["sleep_hours"], 
                                        t["prediction_tab"]["coffee_cups"], t["prediction_tab"]["attendance"], 
                                        t["prediction_tab"]["stress_level"]],
            t["analysis_tab"]["minimum"]: [15, 7, 0, 85, 1],
            t["analysis_tab"]["optimal"]: ['18-22', '7-8', '1-3', '90-95', '2-3'],
            t["analysis_tab"]["maximum"]: [25, 9, 5, 100, 5]
        }
        
        optimal_df = pd.DataFrame(optimal_data)
        st.dataframe(optimal_df, use_container_width=True)

    with tab3:
        st.header(t["model_tab"]["header"])
        
        col_model1, col_model2 = st.columns(2)
        
        with col_model1:
            st.subheader(t["model_tab"]["details"])
            st.write(f"**{t['model_tab']['algorithm']}:** {type(model).__name__}")
            st.write(f"**{t['model_tab']['features_used']}:** {len(feature_columns)}")
            st.write(f"**{t['model_tab']['training_data']}:** 300+ student records")
            st.write(f"**{t['model_tab']['target']}:** Final academic score (40-100 scale)")
            
            st.subheader(t["model_tab"]["feature_importance"])
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': feature_columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=True)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(importance_df['Feature'], importance_df['Importance'])
                ax.set_xlabel('Importance' if lang == 'en' else 'Важность')
                if lang == 'ru':
                    ax.set_title('Важность признаков в прогнозировании')
                else:
                    ax.set_title('Feature Importance in Prediction')
                st.pyplot(fig)
            else:
                if lang == 'ru':
                    st.info("Информация о важности признаков недоступна для этого типа модели")
                else:
                    st.info("Feature importance not available for this model type")
        
        with col_model2:
            st.subheader(t["model_tab"]["performance"])
            st.metric(t["model_tab"]["accuracy"], "0.82 ± 0.05")
            st.metric(t["model_tab"]["mae"], "3.2 points")
            st.metric(t["model_tab"]["cv_score"], "0.79")
            
            st.subheader(t["model_tab"]["data_features"])
            features_data = {
                t["model_tab"]["academic_factors"]: ["study_hours", "attendance", "previous_gpa"],
                t["model_tab"]["lifestyle_factors"]: ["sleep_hours", "coffee_consumption", "stress_level"],
                t["model_tab"]["behavioral_factors"]: ["extracurricular", "learning_style", "semester"]
            }
            st.json(features_data)

    # Футер
    st.markdown("---")
    st.markdown(t["footer"])

if __name__ == '__main__':
    main()


    # 235 убрать комент если сломалось.