import { createUser } from "@/api/create-user";
import { create } from "zustand";
import axios from "axios";
export const useUserStore = create((set) => ({
  user: null,
  isAuthorizing: true,
  isAuthorized: false,

  async authorize() {
    let user = localStorage.getItem("user");

    set({
      isAuthorized: false,
      isAuthorizing: true,
    });

    if (!user) {
      user = await createUser();
    } else {
      user = JSON.parse(user);
    }

    localStorage.setItem("user", JSON.stringify(user));
    
    axios.defaults.headers.common["Authorization"] = user.access_token;

    set({
      isAuthorized: true,
      isAuthorizing: false,
      user,
    });
  },
}));
