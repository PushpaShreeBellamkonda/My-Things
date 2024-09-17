import streamlit as st
import streamlit.components.v1 as components 
from streamlit.components.v1 import html

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

    menu = ['Home', 'Recommend', 'Navigator']
    choice = st.sidebar.selectbox('Menu', menu)

    df = load_data('/home/rguktongole/dataset1.csv')

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
          
       
       
       
         



                    

               
    elif choice == 'Recommend':

        st.subheader('Recommend Courses')
        cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'
                ])
        search_term = st.text_input('Search')
        search_term_up = search_term.upper()
        if st.button('Recommend'):
            if search_term_up == '':
                results = 'Not Found'
                st.warning(results)
            else:
                st.info('Suggested Options include')
                result_df = search_term_if_not_found(search_term_up, df)
                st.dataframe(result_df)
    else:
        st.subheader('Navigator')
        col1,col2=st.columns([1,1])
        with col1:
          im_path="/home/rguktongole/about.jpg"
          width=150
          st.image(im_path,width=width)
        with col2:
          st.write("")
          st.write("")
          def open_html():
             html_file=open("/home/rguktongole/home.html",'r').read()
             html(html_file,width=800,height=600)
          if st.button("open"):
             open_html()
        col3,col4=st.columns([1,1])
        with col3:
          im_path="/home/rguktongole/about.jpg"
          width=150
          st.image(im_path,width=width)
        with col4:
          st.write("")
          st.write("")
          def open_html():
             html_file=open("/home/rguktongole/success.html",'r').read()
             html(html_file,width=800,height=600)
          if st.button("web"):
             open_html()
       
       


if __name__ == '__main__':
    main()

