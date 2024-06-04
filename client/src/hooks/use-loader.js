import { useCallback, useEffect, useReducer } from "react";

const initialState = {
  isLoading: false,
  isLoaded: false,
  isFailed: false,
  error: undefined,
  data: undefined,
};
const reducer = (initialState) => (state, action) => {
  switch (action.type) {
    case "UPDATE":
      return {
        ...state,
        data:
          typeof action.data === "function"
            ? action.data(state.data)
            : action.data,
      };
    case "LOAD":
      return {
        ...initialState,
        isLoading: true,
      };
    case "FAILED":
      return {
        ...state,
        isLoading: false,
        isFailed: true,
        error: action.error,
      };
    case "SUCCESS":
      return {
        ...state,
        isLoaded: true,
        isLoading: false,
        data: action.data,
      };
    case "RESET":
      return initialState;
  }
};

export function useLoader(
  loadFn,
  { autoLoad = false, deps = [], defaultData } = {}
) {
  const reducerInitialState = {
    ...initialState,
    data: defaultData,
  };
  const [state, dispatch] = useReducer(
    reducer(reducerInitialState),
    reducerInitialState
  );

  const load = useCallback(async (...params) => {
    dispatch({ type: "LOAD" });
    try {
      let data = undefined;
      if (typeof params[0] === "function") {
        data = await params?.();
      } else {
        data = await loadFn?.(...params);
      }
      dispatch({ type: "SUCCESS", data });
    } catch (error) {
      dispatch({ type: "FAILED", error });
      console.log(error);
    }
  }, deps);

  useEffect(() => {
    if (autoLoad) {
      load();
    }
  }, [autoLoad, ...deps]);

  return {
    isLoading: state.isLoading,
    isFailed: state.isFailed,
    error: state.error,
    isLoaded: state.isLoaded,
    data: state.data,
    reset: () => dispatch({ type: "RESET" }),
    load,
    update: (data) => dispatch({ type: "UPDATE", data }),
  };
}
