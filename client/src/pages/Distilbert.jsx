import { useEffect, useState, useMemo } from "react";
import Page from "./Page";
import { useData } from "../context/DataContext";
import { useParams } from "react-router-dom";

const names = ["Baseline", "Proposed Solution"];

const Distilbert = () => {
    const { id } = useParams();
    const { data, isSidebarCollapsed } = useData();
    const messages = useMemo(() => data.messages || [], [data]);
    const [botResponse, setBotResponse] = useState(null);

    useEffect(() => {
        const findAndSetBotResponse = () => {
            if (id && messages.length > 0) {
                const message = messages.find((msg) => msg.botResponse?.id == id);
                setBotResponse(message ? message.botResponse : null);
            } else {
                // Raw data when no id is provided
                setBotResponse({
                    categoryResponse: "Raw Category Response Data",
                    text: "Raw Text Data",
                    intentResponse: "Raw Intent Response Data",
                    gradientBoostingCategoryResponse: "Raw Gradient Boosting Category Response Data",
                    gradientBoostingIntentResponse: "Raw Gradient Boosting Intent Response Data",
                });
            }
        };
        findAndSetBotResponse();
    }, [id, messages]);

    return (
        <>
            <div className="relative flex-grow flex flex-col items-center ml-[300px] bg-white h-[670px] w-[1230px] overflow-y-auto p-4"
                style={{
                    width: isSidebarCollapsed ? "1400px" : "1200px",
                    marginLeft: "30px",
                    transition: "all 0.3s ease",
                    scrollbarWidth: "none", // Firefox
                    msOverflowStyle: "none", // Internet Explorer 10+
                }}>
                <div className="mt-[10px] flex flex-col space-y-8">
                    {names.map((name, index) => (
                        <div key={index} className="w-full">
                            {botResponse ? (
                                <Page
                                    name={name}
                                    categoryResponse={
                                        name === "Baseline"
                                            ? "Raw Baseline Category Response"  // Raw data for Baseline
                                            : "Raw Proposed Solution Category Response"  // Raw data for Proposed Solution
                                    }
                                    text={
                                        name === "Baseline"
                                            ? "Raw Baseline Text Data"  // Raw text for Baseline
                                            : "Raw Proposed Solution Text Data"  // Raw text for Proposed Solution
                                    }
                                    intentResponse={
                                        name === "Baseline"
                                            ? "Raw Baseline Intent Response"  // Raw data for Baseline
                                            : "Raw Proposed Solution Intent Response"  // Raw data for Proposed Solution
                                    }
                                    isSidebarCollapsed={isSidebarCollapsed}
                                />
                            ) : (
                                <div className="text-red-500">No data available for {name}</div>
                            )}
                        </div>
                    ))}
                </div>
            </div>
        </>
    );
};

export default Distilbert;
