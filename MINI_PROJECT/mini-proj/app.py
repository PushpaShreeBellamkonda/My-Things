


import streamlit as st
import streamlit.components.v1 as components 
from streamlit.components.v1 import html
import sqlite3


# Load EDA

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel


# Load Our Dataset

def load_data(data):
    df = pd.read_csv(data)
    return df


# Fxn
# Vectorize + Cosine Similarity Matrix

def vectorize_text_to_cosine_mat(data):
    count_vect = CountVectorizer()
    cv_mat = count_vect.fit_transform(data)

    # Get the cosine

    cosine_sim_mat = cosine_similarity(cv_mat)
    return cosine_sim_mat


# Recommendation Sys

@st.cache
def get_recommendation(
    title,
    cosine_sim_mat,
    df,
    num_of_rec=10,
    ):

    # indices of the course

    course_indices = pd.Series(df.index, index=df['course_title'
                               ]).drop_duplicates()

    # Index of course

    idx = course_indices[title]

    # Look into the cosine matr for that index

    sim_scores = list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    selected_course_indices = [i[0] for i in sim_scores[1:]]
    selected_course_scores = [i[0] for i in sim_scores[1:]]

    # Get the dataframe & title

    result_df = df.iloc[selected_course_indices]
    result_df['similarity_score'] = selected_course_scores
    final_recommended_courses = result_df[['course_title', 'url',
            'price', 'num_subscribers']]
    return final_recommended_courses.head(num_of_rec)


# Search For Course

@st.cache_data
def search_term_if_not_found(term, df):
    result_df = df[df['course_title'].str.contains(term)]
    return result_df


def main():
    col1,col2=st.columns([1,4])
    im_path="/home/rguktongole/logo.png"
    with col1:
       st.image(im_path,use_column_width=True)
    with col2:
       st.write("")
       st.markdown("<h1 style='color:#00008B;'>Course Navigator</h1>",unsafe_allow_html=True)

    menu = ['Home', 'Recommend', 'Navigator','Feedback']
    choice = st.sidebar.selectbox('Menu', menu)

    df = load_data('/home/rguktongole/project.csv')

    if choice == 'Home':
       st.write("""<div style="border:5px solid #CADCFC;border-radius:20px;padding:20px;background-color:#00246B;">
       <h3 style="color:#fff;font-size:40px;">About Us</h3>
          <p style="color:#CADCFC;font-family:"Times New Roman",Times,serif;>In today's fast-paced educational landscape, the abundance of available courses can overwhelm learners seeking to enhance their skills or acquire new knowledge. To address this challenge, a personalised course recommendation system is proposed. Leveraging advanced machine learning algorithms and user data, the system tailors recommendations to individual learners, taking into account their preferences and career goals.Through this approach, learners can discover courses aligned with their interests and objectives, facilitating continuous learning and professional development.</p>
          </div>""",unsafe_allow_html=True)
       st.write("")
       st.write("")
       image_path="/home/rguktongole/webd.jpg"
       col1,col2=st.columns([1,4])
       with col1:
          st.write("")
          st.write("")
          st.write("")
          st.image(image_path,use_column_width=True)
       with col2:
          st.write("""<div style="border:5px solid #00246B;border-radius:20px;padding:20px;background-color:#CADCFC;">
          <p style="color:#00246B;font-family:"Times New Roman",Times,serif;>Web development, also known as website development, refers to the tasks associated with creating, building, and maintaining websites and web applications that run online on a browser. It may, however, also include web design, web programming, and database management.</p>
          </div>""",unsafe_allow_html=True)
          st.write("")
          image_path="/home/rguktongole/pythonprog.jpg"
       col3,col4=st.columns([1,4])
       with col3:
          st.write("")
          st.write("")
          st.image(image_path,use_column_width=True)
       with col4:
          st.write("""<div style="border:5px solid #00246B;border-radius:20px;padding:20px;background-color:#CADCFC;">
          <p style="color:#00246B;">Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected.</p>
          </div>""",unsafe_allow_html=True)
          st.write("")
          image_path="/home/rguktongole/javaprogram.jpg"
       col5,col6=st.columns([1,4])
       with col5:
          st.write("")
          st.write("")
          st.image(image_path,use_column_width=True)
       with col6:
          st.write("""<div style="border:5px solid #00246B;border-radius:20px;padding:20px;background-color:#CADCFC;">
          <p style="color:#00246B;">Java is a programming language and computing platform first released by Sun Microsystems in 1995. It has evolved from humble beginnings to power a large share of today's digital world, by providing the reliable platform upon which many services and applications are built</p>
          </div>""",unsafe_allow_html=True)
          st.write("")
          image_path="/home/rguktongole/cprgm.jpg"
       col7,col8=st.columns([1,4])
       with col7:
          st.write("")
          st.write("")
          st.write("")
          st.image(image_path,use_column_width=True)
       with col8:
          st.write("""<div style="border:5px solid #00246B;border-radius:20px;padding:20px;background-color:#CADCFC;">
          <p style="color:#00246B;">C is an imperative procedural language, supporting structured programming, lexical variable scope, and recursion, with a static type system. It was designed to be compiled to provide low-level access to memory and language constructs that map efficiently to machine instructions, all with minimal runtime support.</p>
          </div>""",unsafe_allow_html=True)
          st.write("")
          image_path="/home/rguktongole/cpp.jpg"
       col9,col10=st.columns([1,4])
       with col9:
          st.write("")
          st.write("")
          st.write("")
          st.image(image_path,use_column_width=True)
       with col10:
          st.write("""<div style="border:5px solid #00246B;border-radius:20px;padding:20px;background-color:#CADCFC;">
          <p style="color:#00246B;">C++ is an object-oriented programming (OOP) language that is viewed by many as the best language for creating large-scale applications. C++ is a superset of the C language. A related programming language, Java, is based on C++ but optimized for the distribution of program objects in a network such as the internet.</p>
          </div>""",unsafe_allow_html=True)
          
          st.write("")
          image_path="/home/rguktongole/datascience.jpg"
       col11,col12=st.columns([1,4])
       with col11:
          st.write("")
          st.write("")
          st.write("")
          st.image(image_path,use_column_width=True)
       with col12:
          st.write("""<div style="border:5px solid #00246B;border-radius:20px;padding:20px;background-color:#CADCFC;">
          <p style="color:#00246B;">Data science is the study of data to extract meaningful insights for business. It is a multidisciplinary approach that combines principles and practices from the fields of mathematics, statistics, artificial intelligence, and computer engineering to analyze large amounts of data.</p>
          </div>""",unsafe_allow_html=True)
          
          st.write("")
          image_path="/home/rguktongole/dataanalytics.jpg"
       col13,col14=st.columns([1,4])
       with col13:
          st.write("")
          st.write("")
          st.write("")
          st.image(image_path,use_column_width=True)
       with col14:
          st.write("""<div style="border:5px solid #00246B;border-radius:20px;padding:20px;background-color:#CADCFC;">
          <p style="color:#00246B;">Data analytics is the process of analyzing raw data to find trends and answer questions. It has a broad scope across the field. This process includes many different techniques and goals that can shift from industry to industry.</p>
          </div>""",unsafe_allow_html=True)
          
          st.write("")
          image_path="/home/rguktongole/cloud.jpg"
       col15,col16=st.columns([1,4])
       with col15:
          st.write("")
          st.write("")
          st.write("")
          st.image(image_path,use_column_width=True)
       with col16:
          st.write("""<div style="border:5px solid #00246B;border-radius:20px;padding:20px;background-color:#CADCFC;">
          <p style="color:#00246B;">Cloud computing is the on-demand delivery of IT resources over the Internet with pay-as-you-go pricing. Instead of buying, owning, and maintaining physical data centers and servers, you can access technology services, such as computing power, storage, and databases, on an as-needed basis from a cloud provider like Amazon Web Services (AWS).</p>
          </div>""",unsafe_allow_html=True)
          
          st.write("")
          image_path="/home/rguktongole/ethicalhack.jpg"
       col17,col18=st.columns([1,4])
       with col17:
          st.write("")
          st.write("")
          st.write("")
          st.image(image_path,use_column_width=True)
       with col18:
          st.write("""<div style="border:5px solid #00246B;border-radius:20px;padding:20px;background-color:#CADCFC;">
          <p style="color:#00246B;">Ethical hacking is a process of detecting vulnerabilities in an application, system, or organization's infrastructure that an attacker can use to exploit an individual or organization. They use this process to prevent cyberattacks and security breaches by lawfully hacking into the systems and looking for weak points.</p>
          </div>""",unsafe_allow_html=True)
          
          st.write("")
          image_path="/home/rguktongole/autocad.jpg"
       col19,col20=st.columns([1,4])
       with col19:
          st.write("")
          st.write("")
          st.image(image_path,use_column_width=True)
       with col20:
          st.write("""<div style="border:5px solid #00246B;border-radius:20px;padding:20px;background-color:#CADCFC;">
          <p style="color:#00246B;">AutoCAD is a 2D and 3D computer-aided design (CAD) software application developed by Autodesk. It was first released in December 1982 for the CP/M and IBM PC platforms as a desktop app running on microcomputers with internal graphics controllers.</p>
          </div>""",unsafe_allow_html=True)
          
          st.write("")
          image_path="/home/rguktongole/3ds_max.jpg"
       col21,col22=st.columns([1,4])
       with col21:
          st.write("")
          st.write("")
          st.write("")
          st.image(image_path,use_column_width=True)
       with col22:
          st.write("""<div style="border:5px solid #00246B;border-radius:20px;padding:20px;background-color:#CADCFC;">
          <p style="color:#00246B;">Autodesk 3ds Max, formerly 3D Studio and 3D Studio Max, is a professional 3D computer graphics program for making 3D animations, models, games and images. It is developed and produced by Autodesk Media and Entertainment.</p>
          </div>""",unsafe_allow_html=True)
          
          st.write("")
          image_path="/home/rguktongole/analogelectronics.jpg"
       col23,col24=st.columns([1,4])
       with col23:
          st.write("")
          st.image(image_path,use_column_width=True)
       with col24:
          st.write("""<div style="border:5px solid #00246B;border-radius:20px;padding:20px;background-color:#CADCFC;">
          <p style="color:#00246B;">Analog electronics use continuous signals to represent and process information. These systems are often used in applications where a continuous range of values is required, such as in radio and audio equipment, and in control systems.</p>
          </div>""",unsafe_allow_html=True)
          
          st.write("")
          image_path="/home/rguktongole/communicationsystem.jpg"
       col25,col26=st.columns([1,4])
       with col25:
          st.write("")
          st.image(image_path,use_column_width=True)
       with col26:
          st.write("""<div style="border:5px solid #00246B;border-radius:20px;padding:20px;background-color:#CADCFC;">
          <p style="color:#00246B;">Analog electronics use continuous signals to represent and process information. These systems are often used in applications where a continuous range of values is required, such as in radio and audio equipment, and in control systems.</p>
          </div>""",unsafe_allow_html=True)
          
          st.write("")
          image_path="/home/rguktongole/controlsystems.jpg"
       col27,col28=st.columns([1,4])
       with col27:
          st.write("")
          st.image(image_path,use_column_width=True)
       with col28:
          st.write("""<div style="border:5px solid #00246B;border-radius:20px;padding:20px;background-color:#CADCFC;">
          <p style="color:#00246B;">In a control system, a controller is used to manipulate a variable in order to keep the process functioning at the set point value. The difference between the set point value and the current value of the process variable is called the error, which is used to generate an output signal.</p>
          </div>""",unsafe_allow_html=True)
          
          st.write("")
          image_path="/home/rguktongole/digitalelectronics.jpg"
       col29,col30=st.columns([1,4])
       with col29:
          st.write("")
          st.write("")
          st.image(image_path,use_column_width=True)
       with col30:
          st.write("""<div style="border:5px solid #00246B;border-radius:20px;padding:20px;background-color:#CADCFC;">
          <p style="color:#00246B;">Digital electronics is the foundation of all modern electronic devices such as cellular phones, MP3 players, laptop computers, digital cameras, high definition televisions, etc. Students learn the digital circuit design process to create circuits and present solutions that can improve people's lives.</p>
          </div>""",unsafe_allow_html=True)
          
          st.write("")
          image_path="/home/rguktongole/embededsystems.jpg"
       col31,col32=st.columns([1,4])
       with col31:
          st.write("")
          st.write("")
          st.image(image_path,use_column_width=True)
       with col32:
          st.write("""<div style="border:5px solid #00246B;border-radius:20px;padding:20px;background-color:#CADCFC;">
          <p style="color:#00246B;">An embedded system is a combination of computer hardware and software designed for a specific function. Embedded systems may also function within a larger system. The systems can be programmable or have a fixed functionality.</p>
          </div>""",unsafe_allow_html=True)
          
          st.write("")
          image_path="/home/rguktongole/etabs.jpg"
       col33,col34=st.columns([1,4])
       with col33:
          st.write("")
          st.write("")
          st.image(image_path,use_column_width=True)
       with col34:
          st.write("""<div style="border:5px solid #00246B;border-radius:20px;padding:20px;background-color:#CADCFC;">
          <p style="color:#00246B;">ETABS is an engineering software product that caters to multi-story building analysis and design. Modeling tools and templates, code-based load prescriptions, analysis methods and solution techniques, all coordinate with the grid-like geometry unique to this class of structure.</p>
          </div>""",unsafe_allow_html=True)
          
          st.write("")
          image_path="/home/rguktongole/iot.jpg"
       col35,col36=st.columns([1,4])
       with col35:
          st.write("")
          st.write("")
          st.write("")
          st.image(image_path,use_column_width=True)
       with col36:
          st.write("""<div style="border:5px solid #00246B;border-radius:20px;padding:20px;background-color:#CADCFC;">
          <p style="color:#00246B;">The Internet of Things (IoT) describes the network of physical objects—“things”—that are embedded with sensors, software, and other technologies for the purpose of connecting and exchanging data with other devices and systems over the internet.</p>
          </div>""",unsafe_allow_html=True)
          
          st.write("")
          image_path="/home/rguktongole/networking.jpg"
       col37,col38=st.columns([1,4])
       with col37:
          st.write("")
          st.write("")
          st.write("")
          st.image(image_path,use_column_width=True)
       with col38:
          st.write("""<div style="border:5px solid #00246B;border-radius:20px;padding:20px;background-color:#CADCFC;">
          <p style="color:#00246B;">Networking is the process of making connections and building relationships. These connections can provide you with advice and contacts, which can help you make informed career decisions. Networking can even help you ﬁnd unadvertised jobs/internships. Networking can take place in a group or one-on-one setting.</p>
          </div>""",unsafe_allow_html=True)
          
          st.write("")
          image_path="/home/rguktongole/primaverap6.jpg"
       col39,col40=st.columns([1,4])
       with col39:
          st.write("")
          st.image(image_path,use_column_width=True)
       with col40:
          st.write("""<div style="border:5px solid #00246B;border-radius:20px;padding:20px;background-color:#CADCFC;">
          <p style="color:#00246B;">Oracle Primavera P6 is a project management software used to plan, schedule, and manage complex projects. This software is widely used in industries such as construction, engineering, and infrastructure, to name a few.</p>
          </div>""",unsafe_allow_html=True)
          
          st.write("")
          image_path="/home/rguktongole/robotics.jpg"
       col40,col41=st.columns([1,4])
       with col40:
          st.write("")
          st.write("")
          st.image(image_path,use_column_width=True)
       with col41:
          st.write("""<div style="border:5px solid #00246B;border-radius:20px;padding:20px;background-color:#CADCFC;">
          <p style="color:#00246B;">Robotics is a branch of engineering and computer science that involves the conception, design, manufacture and operation of robots. The objective of the robotics field is to create intelligent machines that can assist humans in a variety of ways. Robotics can take on a number of forms.</p>
          </div>""",unsafe_allow_html=True)
          
          st.write("")
          image_path="/home/rguktongole/vlsi.jpg"
       col42,col43=st.columns([1,4])
       with col42:
          st.write("")
          st.write("")
          st.image(image_path,use_column_width=True)
       with col43:
          st.write("""<div style="border:5px solid #00246B;border-radius:20px;padding:20px;background-color:#CADCFC;">
          <p style="color:#00246B;">VLSI (Very Large-Scale Integration) design is a process of designing integrated circuits (ICs) by integrating thousands, millions or even billions of transistors on a single chip. These ICs are used in a variety of electronic devices ranging from simple handheld devices to complex supercomputers.</p>
          </div>""",unsafe_allow_html=True)
          
          st.write("")
          image_path="/home/rguktongole/staadpro.jpg"
       col44,col45=st.columns([1,4])
       with col44:
          st.write("")
          st.write("")
          st.image(image_path,use_column_width=True)
       with col45:
          st.write("""<div style="border:5px solid #00246B;border-radius:20px;padding:20px;background-color:#CADCFC;">
          <p style="color:#00246B;">STAAD's full form is Structural Analysis and Design. STAAD Pro is one of the popular software that is used for analyzing & designing structures like – buildings, towers, bridges, industrial, transportation, and utility structures.</p>
          </div>""",unsafe_allow_html=True)
          
          st.write("")
          image_path="/home/rguktongole/rivitarchitecture.jpg"
       col46,col47=st.columns([1,4])
       with col46:
          st.write("")
          st.write("")
          st.image(image_path,use_column_width=True)
       with col47:
          st.write("""<div style="border:5px solid #00246B;border-radius:20px;padding:20px;background-color:#CADCFC;">
          <p style="color:#00246B;">From the outset, Revit was intended to allow architects and other building professionals to design and document a building by creating a parametric three-dimensional model that included both the geometry and non-geometric design and construction information, which is also known as building information modeling or BIM </p>
          </div>""",unsafe_allow_html=True)
          
          
          
                  
    elif choice == 'Recommend':

        st.subheader('Recommend Courses')
        cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'
                ])
        search_term = st.text_input('Search')
        search_term_up = search_term.upper()
        if st.button('Recommend'):
            if search_term_up == '':
                results = 'Enter any Course name'
                st.warning(results)
            else:
                st.info('Suggested Options include')
                result_df = search_term_if_not_found(search_term_up, df)
                st.dataframe(result_df)
    elif choice=='Navigator':
        st.header('Navigator')
        im_path="/home/rguktongole/webd.jpg"
        width=500
        st.image(im_path,width=width)
        st.write("")
        html_url = "http://localhost:8000/WT_nav.html"
        st.markdown(f'<button><a href="{html_url}" target="_blank" style="text-decoration:none;font-size:20px;color:#00246B;background-color:#CADCFC;">Web Development</a></button>', unsafe_allow_html=True)
 
        im_path="/home/rguktongole/pythonprog.jpg"
        width=500
        st.image(im_path,width=width)
        st.write("")
        html_url = "http://localhost:8000/python.html"
        st.markdown(f'<button><a href="{html_url}" target="_blank" style="text-decoration:none;font-size:20px;color:#00246B;background-color:#CADCFC;">Python Programming</a></button>', unsafe_allow_html=True)
        
        im_path="/home/rguktongole/javaprogram.jpg"
        width=500
        st.image(im_path,width=width)
        st.write("")
        html_url = "http://localhost:8000/java.html"
        st.markdown(f'<button><a href="{html_url}" target="_blank" style="text-decoration:none;font-size:20px;color:#00246B;background-color:#CADCFC;">Java Programming</a></button>', unsafe_allow_html=True)
        
        im_path="/home/rguktongole/datascience.jpg"
        width=500
        st.image(im_path,width=width)
        st.write("")
        html_url = "http://localhost:8000/dscience.html"
        st.markdown(f'<button><a href="{html_url}" target="_blank" style="text-decoration:none;font-size:20px;color:#00246B;background-color:#CADCFC;">Data Science</a></button>', unsafe_allow_html=True)
        
        im_path="/home/rguktongole/ml.jpg"
        width=500
        st.image(im_path,width=width)
        st.write("")
        html_url = "http://localhost:8000/ml_nav.html"
        st.markdown(f'<button><a href="{html_url}" target="_blank" style="text-decoration:none;font-size:20px;color:#00246B;background-color:#CADCFC;">Machine Learning</a></button>', unsafe_allow_html=True)
        
        im_path="/home/rguktongole/ethicalhack.jpg"
        width=500
        st.image(im_path,width=width)
        st.write("")
        html_url = "http://localhost:8000/ethicalhack_nav.html"
        st.markdown(f'<button><a href="{html_url}" target="_blank" style="text-decoration:none;font-size:20px;color:#00246B;background-color:#CADCFC;">Ethical Hacking</a></button>', unsafe_allow_html=True)
        
        im_path="/home/rguktongole/cloud.jpg"
        width=500
        st.image(im_path,width=width)
        st.write("")
        html_url = "http://localhost:8000/cloud_nav.html"
        st.markdown(f'<button><a href="{html_url}" target="_blank" style="text-decoration:none;font-size:20px;color:#00246B;background-color:#CADCFC;">Cloud Computing</a></button>', unsafe_allow_html=True)
        
        im_path="/home/rguktongole/vlsinew.jpg"
        width=500
        st.image(im_path,width=width)
        st.write("")
        html_url = "http://localhost:8000/vlsi.html"
        st.markdown(f'<button><a href="{html_url}" target="_blank" style="text-decoration:none;font-size:20px;color:#00246B;background-color:#CADCFC;">VLSI</a></button>', unsafe_allow_html=True)
        
        im_path="/home/rguktongole/embededsystems.jpg"
        width=500
        st.image(im_path,width=width)
        st.write("")
        html_url = "http://localhost:8000/esystems.html"
        st.markdown(f'<button><a href="{html_url}" target="_blank" style="text-decoration:none;font-size:20px;color:#00246B;background-color:#CADCFC;">Embedded Systems</a></button>', unsafe_allow_html=True)
        
        im_path="/home/rguktongole/robotics.jpg"
        width=500
        st.image(im_path,width=width)
        st.write("")
        html_url = "http://localhost:8000/robotics_nav.html"
        st.markdown(f'<button><a href="{html_url}" target="_blank" style="text-decoration:none;font-size:20px;color:#00246B;background-color:#CADCFC;">Robotics</a></button>', unsafe_allow_html=True)
        
    else:
        def create_table():
           conn = sqlite3.connect('feedback.db')
           c = conn.cursor()
           c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT, feedback TEXT)''')
           conn.commit()
           conn.close()

# Function to insert feedback into the database
        def insert_feedback(name, email, feedback):
          conn = sqlite3.connect('feedback.db')
          c = conn.cursor()
          c.execute("INSERT INTO feedback (name, email, feedback) VALUES (?, ?, ?)", (name, email, feedback))
          conn.commit()
          conn.close()

# Main function
        def fun():
          st.title("Feedback Form")

    # Create database table if it doesn't exist
          create_table()

    # Input fields for feedback
          name = st.text_input("Name")
          email = st.text_input("Email")
          feedback = st.text_area("Feedback")

    # Submit button
          if st.button("Submit"):
            if name and email and feedback:
               insert_feedback(name, email, feedback)
               st.success("Feedback submitted successfully!")
            else:
               st.error("Please fill in all fields.")

        if __name__ == "__main__":
               fun()
       
       
        
if __name__ == '__main__':
    main()

