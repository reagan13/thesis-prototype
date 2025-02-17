// Distilbert.jsx
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
            if (!id || messages.length === 0) return;
            const message = messages.find((msg) => msg.botResponse?.id == id);
            setBotResponse(message ? message.botResponse : null);
        };
        findAndSetBotResponse();
    }, [id, messages]);

    return (
        <>
            {!id ? (
                <div className="flex justify-center items-center h-full w-full text-white">
                    <h2>Please submit a message to view the results</h2>
                </div>
            ) : (
                <div className="relative flex-grow flex flex-col items-center ml-[300px] bg-black h-[680px] w-[1230px] overflow-y-auto p-4"
				style={{
					width: isSidebarCollapsed ? "1400px" : "1230px",
					marginLeft: isSidebarCollapsed ? "110px" : "300px",
					transition: "all 0.3s ease",
					scrollbarWidth: "none", // Firefox
					msOverflowStyle: "none", // Internet Explorer 10+
				}}>
                    <div className="mt-[10px] flex flex-col space-y-10">
                        {names.map((name, index) => (
                            <div key={index} className="w-full">
                                {botResponse ? (
                                    <Page
                                        name={name}
                                        categoryResponse={
                                            name === "Baseline"
                                                ? botResponse.categoryResponse
                                                : botResponse.gradientBoostingCategoryResponse
                                        }
                                        text={botResponse.text}
                                        intentResponse={
                                            name === "Baseline"
                                                ? botResponse.intentResponse
                                                : botResponse.gradientBoostingIntentResponse
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
            )}
        </>
    );
};

export default Distilbert;
