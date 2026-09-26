import { formatUsd } from "../../format";
import { useAnimatedNumber } from "../../format";

export default function AnimatedNumber({ value, format = formatUsd, className }) {
  const display = useAnimatedNumber(value);
  return <span className={className}>{format(display)}</span>;
}
