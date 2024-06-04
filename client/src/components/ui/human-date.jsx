import { formatDistance } from "date-fns";
import { useEffect, useState } from "react";

export const HumanDate = ({ date, ...rest }) => {
  const [currentDate, setCurrentDate] = useState(() => new Date());

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentDate(new Date());
    }, 10000);

    return () => {
      clearInterval(interval);
    };
  }, []);
  const a = formatDistance(date, currentDate, {
    addSuffix: true,
  });

  return <span {...rest}>{a}</span>;
};
