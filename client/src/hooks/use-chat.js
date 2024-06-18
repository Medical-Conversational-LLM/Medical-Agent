import { useState } from "react";
import { getChat } from "@/api/get-chat";
import { postMessage } from "@/api/post-message";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { nanoid } from "nanoid";

const createMessage = (content, me = false) => {
  return {
    id: nanoid(),
    content,
    me,
    created_at: new Date(),
  };
};

const getChatQueryKey = (id) => ["chat", String(id)];

export function useChat(chatId) {
  const [messageDict, setMessageDict] = useState({});
  const [lastMessageState, setLastMessageState] = useState();

  const [settings, setSettings] = useState({
    temperature: 1,
    max_length: 256,
    top_k: 1,
    top_p: 1,
  });

  const queryClient = useQueryClient();

  const query = useQuery({
    queryKey: getChatQueryKey(chatId),
    queryFn: () => getChat(chatId),
    staleTime: 100000,
    enabled: !chatId ? false : undefined,
  });

  const mutation = useMutation({
    mutationFn: postMessage,
    mutationKey: getChatQueryKey(chatId),
  });

  const [isSendingDict, setIsSendingDict] = useState({});

  const navigate = useNavigate();

  const addMessages = (messages = []) => {
    queryClient.setQueryData(getChatQueryKey(chatId), (data) => ({
      ...(data || {}),
      messages: [...(data?.messages || []), ...messages],
    }));
  };

  const updateMessages = (chatId, messages) => {
    const dict = messages.reduce((acc, message) => {
      acc = {
        ...acc,
        [message.id]: {
          ...message,
        },
      };

      if (message.oldId) {
        acc[message.oldId] = message;
      }
      return acc;
    }, {});
    

    console.log("chatId", chatId)

    queryClient.setQueryData(getChatQueryKey(chatId), (data) => { 

      console.log("UPDATING" ,dict, data)
      const final = {
        ...data,
        messages: data.messages.map((message) => {
          if (!dict[message.id]) {
            return message;
          }

          return {
            ...message,
            ...dict[message.id],
          };
        }),
      }

      console.log("UPDATED", final)
      return final;
    });
  };

  const onSubmit = async () => {
    const messageText = messageDict[chatId];
    if (messageText.trim() === "") {
      return;
    }
    setIsSendingDict((dict) => ({ ...dict, [chatId]: true }));

    const myMessage = createMessage(messageText, true);

    let botMessage = createMessage("", false);

    addMessages([myMessage, botMessage]);

    await mutation.mutateAsync({
      chatId,
      message: myMessage,
      settings,

      onMessage(message) {
        console.log("message", message);

        setLastMessageState(undefined);

        if (!message || typeof message !== "object") {
          return;
        }
        switch (message.type) {
          case "NEW_CHAT":
            message = message.chat;
            queryClient.setQueryData(["chats"], (chats) => [
              {
                title: message.title,
                id: message.id,
              },
              ...chats,
            ]);
 
            botMessage = {
              ...botMessage,
              oldId: botMessage.id,
              ...message.response_message,
            };

            queryClient.setQueryData(
              getChatQueryKey(message.id),
              queryClient.getQueryData(getChatQueryKey(chatId))
            );

            setMessageDict((dict) => ({
              ...deleteFromDict(dict, undefined),
              [message.id]: myMessage.content,
            }));

            navigate(`/chat/${message.id}`);

            queryClient.setQueryData(getChatQueryKey(chatId), null);

            setIsSendingDict((dict) => ({
              ...dict,
              [undefined]: false,
              [message.id]: true,
            }));

            chatId = message.id;

            break;
          case "TOKEN":
            botMessage = {
              ...botMessage,
              content: botMessage.content + message.message,
            };
            console.log("botMessage", botMessage)
            updateMessages(chatId, [botMessage]);
            break;
          default:
            setLastMessageState(message);
            break;
        }
      },
    });
    setIsSendingDict((dict) => ({ ...dict, [chatId]: false }));
    setMessageDict((dict) => deleteFromDict(dict, chatId, undefined));
  };
 

  return {
    onSubmit,
    data: query.data,
    setMessage: (message) =>
      setMessageDict((dict) => ({
        ...dict,
        [chatId]: message,
      })),
    message: messageDict[chatId] || "",
    addMessages,
    isLoading: query.isLoading,
    isSending: isSendingDict[chatId] === true,
    chatId,
    lastMessageState,
    settings,
    updateSetting(name, value) {
      setSettings((settings) => ({ ...settings, [name]: value }));
    },
  };
}

const deleteFromDict = (dict, ...keys) => {
  for (let key of keys) {
    // eslint-disable-next-line no-unused-vars
    const { [key]: toDelete, ...rest } = dict;
    dict = rest;
  }

  return dict;
};
