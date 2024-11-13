import streamlit as st

st.set_page_config(
    page_title="PANDAWA - Pendeteksi Awal Penyakit Mata",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("<h1 style='text-align: center; color: #007ACC;'>PANDAWA</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #007ACC;'>Pendeteksi Awal Penyakit Mata</h2>", unsafe_allow_html=True)
st.subheader("Penerapan Model ML EfficientNet untuk Deteksi Dini Penyakit Mata")

st.write("""
<div style='text-align: left; font-size: 18px;'>
Selamat datang di Aplikasi Klasifikasi Penyakit Mata! Kami hadir untuk membantu deteksi dini kondisi kesehatan mata menggunakan teknologi machine learning. Aplikasi ini mampu mengklasifikasikan gambar retina dalam empat kategori:
</div>
""", unsafe_allow_html=True)
st.write("")

st.markdown("""
<ul style="font-size: 18px;">
<li><strong>Glaukoma</strong></li>
<li><strong>Retinopati Diabetik</strong></li>
<li><strong>Katarak</strong></li>
<li><strong>Normal</strong></li>
</ul>
""", unsafe_allow_html=True)

st.markdown("<div style='text-align: left; font-size: 18px;'>Deteksi dini ini penting untuk mencegah kebutaan dan meningkatkan kualitas hidup.</div>", unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["Latar Belakang", "Deteksi Katarak", "Deteksi Glaukoma", "Deteksi Retinopati Diabetik"])

with tab1:
    
    st.markdown("""
    > <i>“Early detection and treatment of eye diseases such as cataracts, diabetic retinopathy, and glaucoma are crucial in preventing vision loss. Regular eye examinations can identify these conditions before significant damage occurs, allowing for timely intervention and better outcomes.”</i>
    > 
    > — Dr. Paul Sieving, former Director of the National Eye Institute (2019)
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("## Latar Belakang Penyakit Mata di Indonesia")
        st.write("""
        - **Glaukoma**: Menurut laporan Kementerian Kesehatan (2019), prevalensi glaukoma di Indonesia diperkirakan mencapai 0,46% (setiap 4-5 orang per 1.000 penduduk). Secara global, jumlah penderita glaukoma pada 2020 mencapai 76 juta orang, meningkat dari 60,5 juta satu dekade sebelumnya.
        - **Retinopati Diabetik**: Di Indonesia, 42,6% penderita diabetes mengalami retinopati diabetik, dan 10% diantaranya mengalami kebutaan (Dinkes DIY, 2024). Diabetes juga meningkatkan risiko glaukoma dan katarak, namun sebagian besar penderita tidak melakukan pemeriksaan mata rutin.
        - **Katarak**: Menjadi penyebab utama kebutaan, dengan lebih dari 1,2 juta kasus baru setiap tahunnya (WHO, 2022), katarak menyumbang 75% dari kasus kebutaan di Indonesia.
        """)
    with col2:
        st.image('assets/image.png', caption='Contoh Gambar Retina', use_container_width=True)

    st.write("## Potensi Machine Learning untuk Deteksi Dini")
    st.write("""
    Dengan kemampuan untuk menganalisis gambar retina secara otomatis, machine learning dapat membantu mendeteksi perubahan yang menunjukkan adanya penyakit mata sebelum gejala lanjut muncul. Deteksi dini ini penting untuk mengurangi risiko kebutaan pada pasien dan memungkinkan tindakan medis segera.
    """)

with tab2:
    # Deteksi Katarak Tab
    col5, col6 = st.columns([1, 1])
    with col5:
        st.write("## Bagaimana Katarak Dideteksi")
        st.write("""
        Katarak dapat dideteksi melalui analisis gambar retina yang menunjukkan kekeruhan pada lensa mata. Model machine learning mempelajari pola ini dari gambar retina untuk mengidentifikasi kasus katarak.
        """)
    with col6:
        st.image('assets/image.png', caption='Ilustrasi Katarak', use_container_width=True)

with tab3:
    # Deteksi Glaukoma Tab
    col7, col8 = st.columns([1, 1])
    with col7:
        st.write("## Bagaimana Glaukoma Dideteksi")
        st.write("""
        Glaukoma ditandai dengan peningkatan rasio antara Optic Cup (OC) dan Optic Disc (OD), yang menyebabkan kerusakan pada saraf optik. Deteksi dini perubahan ini sangat penting untuk mencegah kebutaan.
        """)
    with col8:
        st.image('assets/image.png', caption='Ilustrasi Glaukoma', use_container_width=True)

with tab4:
    # Deteksi Retinopati Diabetik Tab
    col9, col10 = st.columns([1, 1])
    with col9:
        st.write("## Bagaimana Retinopati Diabetik Dideteksi")
        st.write("""
        Retinopati diabetik dapat dikenali melalui tanda-tanda seperti pembuluh darah yang bocor dan kerusakan retina. Machine learning dapat mendeteksi pola ini dalam gambar retina dan membantu diagnosis dini.
        """)
    with col10:
        st.image('assets/image.png', caption='Ilustrasi Retinopati Diabetik', use_container_width=True)

st.markdown("<hr style='border: 1px solid #007ACC;'>", unsafe_allow_html=True)
st.write("Jelajahi aplikasi kami untuk mendeteksi dini penyakit mata.")
