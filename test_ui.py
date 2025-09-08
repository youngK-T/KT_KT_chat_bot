# Gradio 인터페이스 구성
with gr.Blocks() as demo:
    session_state = gr.State(initialize_state())

    gr.Markdown("# 🤖 AI 면접관 \n이력서를 업로드하고 인터뷰를 시작하세요!")

    with gr.Row():
        file_input = gr.File(label="이력서 업로드 (PDF 또는 DOCX)")
        upload_btn = gr.Button("인터뷰 시작")

    chatbot = gr.Chatbot()
    user_input = gr.Textbox(show_label=False, placeholder="답변을 입력하고 Enter를 누르세요.")

    upload_btn.click(upload_and_initialize, inputs=[file_input, session_state], outputs=[session_state, chatbot])
    user_input.submit(chat_interview, inputs=[user_input, session_state], outputs=[session_state, chatbot])
    user_input.submit(lambda: "", None, user_input)

# 실행
demo.launch(share=True)