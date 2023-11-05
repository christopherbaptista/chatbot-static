class Chatbox {
  constructor() {
    this.args = {
      openButton: document.querySelector(".chatbox__button"),
      chatBox: document.querySelector(".chatbox__support"),
      sendButton: document.querySelector(".send__button"),
      questionDropdown: document.querySelector("#questionDropdown"), // Menambahkan referensi ke elemen dropdown
    };

    this.state = false;
    this.messages = [];
  }

  display() {
    const { openButton, chatBox, sendButton, questionDropdown } = this.args;

    openButton.addEventListener("click", () => this.toggleState(chatBox));

    sendButton.addEventListener("click", () => this.onSendButton(chatBox));

    const node = chatBox.querySelector("input");
    node.addEventListener("keyup", ({ key }) => {
      if (key === "Enter") {
        this.onSendButton(chatBox);
      }
    });

    // Handle perubahan dropdown
    questionDropdown.addEventListener("change", () => {
      this.onQuestionSelect(questionDropdown, node);
    });
  }

  toggleState(chatbox) {
    this.state = !this.state;

    // Menampilkan atau menyembunyikan kotak chat
    if (this.state) {
      chatbox.classList.add("chatbox--active");
    } else {
      chatbox.classList.remove("chatbox--active");
    }
  }

  onSendButton(chatbox) {
    var textField = chatbox.querySelector("input");
    let text1 = textField.value;
    if (text1 === "") {
      return;
    }

    let msg1 = { name: "User", message: text1 };
    this.messages.push(msg1);

    fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: JSON.stringify({ message: text1 }),
      mode: "cors",
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then((response) => response.json())
      .then((response) => {
        let msg2 = { name: "Chatbot APADOK", message: response.answer };
        this.messages.push(msg2);
        this.updateChatText(chatbox);
        textField.value = "";
      })
      .catch((error) => {
        console.error("Error:", error);
        this.updateChatText(chatbox);
        textField.value = "";
      });
  }

  updateChatText(chatbox) {
    var html = "";
    this.messages
      .slice()
      .reverse()
      .forEach(function (item, index) {
        if (item.name === "Chatbot APADOK") {
          html += `<div class="messages__item messages__item--visitor">${item.message}</div>`;
        } else {
          html += `<div class="messages__item messages__item--operator">${item.message}</div>`;
        }
      });

    const chatmessage = chatbox.querySelector(".chatbox__messages");
    chatmessage.innerHTML = html;
  }

  onQuestionSelect(questionDropdown, inputNode) {
    const selectedQuestion = questionDropdown.value;

    // Memasukkan pertanyaan yang dipilih ke dalam kotak input
    inputNode.value = selectedQuestion;

    // Fokuskan ke kotak input
    inputNode.focus();
  }
}

const chatbox = new Chatbox();
chatbox.display();
