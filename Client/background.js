// Function to send a message to the content script
function sendMessageToContentScript() {
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
        if (tabs.length > 0) {
            var tab = tabs[0];
            console.log("hello",tab.url);
            chrome.tabs.sendMessage(tab.id, { message: "Hello from background.js!", url: tab.url });
        }
    });
}

// Add listener for onInstalled event
chrome.runtime.onInstalled.addListener(function () {
    sendMessageToContentScript();
});

// Add listener for onUpdated event
chrome.tabs.onUpdated.addListener(function (tabId, changeInfo, tab) {
    if (changeInfo.status === "complete" && tab.active) {
        sendMessageToContentScript();
    }
});