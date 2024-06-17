## Front-End Architecture Overview
### ⚛️ [React](https://react.dev/)

The front-end architecture of the application is designed to provide a cohesive and interactive chat interface for users. Here’s an overview of its key components and functionalities:

### Component Structure

- **Chat Component (`Chat`)**: This component serves as the primary interface for users to interact with the chat system. It includes:
  - **Message List**: Displays chat messages fetched from the server, updating dynamically as new messages arrive.
  - **Message Input**: Allows users to type and send messages to the chat. It manages user inputs and initiates message sending.
  - **Suggestions**: Provides predefined prompts to facilitate quick message generation.

- **Layout Component (`Layout`)**: Wraps the chat interface (`Chat` component) with consistent styling and layout configurations, ensuring uniform appearance across different application sections.

### State Management

- **React State (`useState`)**: Manages local component state for critical aspects such as message data (`messageDict`), message sending states (`isSendingDict`), chat settings (`settings`), and last message state (`lastMessageState`). This state management facilitates real-time updates and user interaction feedback.

### Effect Hooks

- **Scroll Management**: Utilizes `useRef` to maintain references to the message list (`listRef`) and input field (`inputRef`). `useEffect` hooks automatically scroll to the latest message when new messages arrive or when the chat is loading.

### Data Fetching and Mutation

- **React Query (`useQuery`, `useMutation`, `useQueryClient`)**: Integrates with server-side APIs (`getChat`, `postMessage`) to fetch chat data (`useQuery`) and send messages (`useMutation`). It manages data caching (`queryClient`) and updates the UI based on server responses.

### Navigation

- **React Router DOM (`useNavigate`)**: Facilitates navigation between different chat sessions based on user interactions (`navigate` function), ensuring smooth transitions between chat rooms or threads.

### Component Integration and Interaction

- **Component Communication**: Uses props (`chats`, `message`) to pass data and callbacks (`setMessage`, `onSubmit`) between parent (`Layout`) and child components (`ChatInput`, `ChatMessagesList`, `ChatSuggestions`). This facilitates seamless interaction and updates within the chat interface.

### Overall User Experience

- **Responsive Design**: Ensures the chat interface adapts to various screen sizes (`flex` and `overflow` properties), enhancing usability across different devices.
