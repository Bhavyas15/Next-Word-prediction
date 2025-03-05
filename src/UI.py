import streamlit as st
from functions import space
from functions import predict_next_word

def ui_ux(model,tokenizer):
    st.title('✨ Next Word Predictor ✨')
    st.subheader('Type Your Text & Let AI Predict the Next Words!🚀🚀🚀🚀🚀')

    space(3)

    # Initialize session state for text if not set
    if "text" not in st.session_state:
        st.session_state['text'] = "This is"

    def update_session_text():
        st.session_state["text"] = st.session_state["text_input"]

    # Function to append selected word to text
    def update_text(word):
        st.session_state.text += " " + word  # Append selected word

    # Text input with on_change callback
    st.markdown("<h4 style='color: orange;'>📝 Enter Input Text :</h4>", unsafe_allow_html=True)
    text = st.text_input("", 
                value=st.session_state.text, 
                key="text_input", 
                on_change=update_session_text)

    _,_,pred_top_words = predict_next_word(model, tokenizer, text)

    # print(pred_top_words[len(pred_top_words)-1])
    top_words=pred_top_words[len(pred_top_words)-1]
    # print(top_words.split())
    top_words=top_words.split()

    print(text)

    if text.strip():
        space(2)
        # st.write(" 🔮 Predicted Next Words:")
        st.markdown("<h4 style='color: pink;'>🔮 Predicted Next Words :</h4>", unsafe_allow_html=True)
        cols = st.columns(3)

        for idx,word in enumerate(top_words):
            with cols[idx%3]:
                st.button(word, key=word, on_click=update_text, args=(word,))  # Callback without rerun