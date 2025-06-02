document.addEventListener('DOMContentLoaded', () => {
    const chatLog = document.getElementById('chat-log');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const newChatButton = document.getElementById('newChatButton');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const backendUrl = '/api/converse';

    let sessionId = null;
    let lastQuestionText = null;
    let isFirstMessage = true;

    function showLoading(show) {
        if (show) {
            loadingIndicator.classList.remove('hidden');
            sendButton.disabled = true;
            userInput.disabled = true;
        } else {
            loadingIndicator.classList.add('hidden');
            sendButton.disabled = false;
            userInput.disabled = false;
            userInput.focus();
        }
    }

    function resetChat() {
        chatLog.innerHTML = '';
        sessionId = null;
        lastQuestionText = null;
        isFirstMessage = true;
        userInput.disabled = false;
        sendButton.disabled = false;
        userInput.placeholder = "Enter your vibe or answer here...";
        appendMessage("Hello! What kind of vibe are you looking for today?", 'agent');
    }

    function appendMessage(text, sender, type = '') {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender === 'user' ? 'user-message' : 'agent-message');
        if (sender === 'agent' && type) {
            messageDiv.classList.add(type);
        }
        
        const textNode = document.createTextNode(text);
        const contentDiv = document.createElement('div');
        contentDiv.appendChild(textNode);
        messageDiv.innerHTML = contentDiv.innerHTML.replace(/\n/g, '<br>');

        chatLog.appendChild(messageDiv);
        chatLog.scrollTop = chatLog.scrollHeight;
    }
    
    function displayProducts(products, justification) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', 'agent-message', 'product-recommendation');
        
        let htmlContent = '';
        if (justification) {
            htmlContent += `<strong>Justification:</strong><p>${escapeHtml(justification).replace(/\n/g, '<br>')}</p>`;
        }

        if (products && products.length > 0) {
            htmlContent += '<strong>Recommended Products:</strong><ul>';
            products.forEach(product => {
                htmlContent += `<li>
                    <strong>${escapeHtml(product.name || 'N/A')}</strong><br>
                    Category: ${escapeHtml(product.category || 'N/A')}<br>
                    Price: $${escapeHtml(String(product.price || 'N/A'))}<br>
                    Fit: ${escapeHtml(product.fit || 'N/A')}<br>
                    Sizes: ${escapeHtml(product.available_sizes || 'N/A')}
                </li>`;
            });
            htmlContent += '</ul>';
        } else if (!justification) {
            htmlContent += '<p>No products found matching your criteria.</p>';
        }
        
        messageDiv.innerHTML = htmlContent;
        chatLog.appendChild(messageDiv);
        chatLog.scrollTop = chatLog.scrollHeight;
    }

    function escapeHtml(unsafe) {
        if (unsafe === null || typeof unsafe === 'undefined') return '';
        return String(unsafe)
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }


    async function sendMessage() {
        const messageText = userInput.value.trim();
        if (!messageText) return;

        appendMessage(messageText, 'user');
        userInput.value = '';
        showLoading(true);

        const payload = {};
        if (sessionId) {
            payload.session_id = sessionId;
        }

        if (isFirstMessage) {
            payload.vibe_description = messageText;
            isFirstMessage = false;
            userInput.placeholder = "Your answer...";
        } else {
            payload.user_response = messageText;
            if (lastQuestionText) {
                payload.last_question_text = lastQuestionText;
            }
        }
        
        try {
            const response = await fetch(backendUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: "Unknown server error" }));
                throw new Error(`Server error: ${response.status} - ${errorData.error || response.statusText}`);
            }

            const data = await response.json();
            
            sessionId = data.session_id;
            lastQuestionText = data.follow_up_question;

            if (data.follow_up_question) {
                appendMessage(data.follow_up_question, 'agent');
            }

            if (data.products || (data.justification && !data.follow_up_question)) {
                displayProducts(data.products, data.justification);
                if (!data.follow_up_question) {
                     userInput.disabled = true;
                     sendButton.disabled = true;
                     userInput.placeholder = "Conversation ended.";
                }
            } else if (data.justification && !data.follow_up_question && !data.products) {
                appendMessage(data.justification, 'agent');
                userInput.disabled = true;
                sendButton.disabled = true;
                userInput.placeholder = "Conversation ended.";
            }


        } catch (error) {
            console.error('Error sending message:', error);
            appendMessage(`Error: ${error.message}`, 'agent', 'error-message');
        } finally {
            showLoading(false);
        }
    }

    newChatButton.addEventListener('click', resetChat);
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    resetChat();
});
